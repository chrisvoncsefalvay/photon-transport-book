import {
  assetUrl,
  decodePlanes,
  decodeFiniteDifferenceStatus,
  formatValue,
  parameterLabel,
  poseLabel,
  pixelIndex,
  PARAMETERS,
  validateData,
  validateMeshes,
  unresolvedCoordinates,
  type Parameter,
  type RecordedPose,
} from "./data";
import {
  committedCoordinates,
  deltaPixels,
  DisplayedProjectionHistory,
} from "./history";
import { steppedRecordedPose } from "./interaction";
import type { AnatomyViewer } from "./viewer";

const ELEMENT = "dpt-pose-sensitivity";

function mountFigure(root: HTMLElement): () => void {
  const data = validateData(JSON.parse(root.dataset.record ?? "null"));
  const query = <E extends HTMLElement>(selector: string) => {
    const element = root.querySelector<E>(selector);
    if (!element) throw new Error(`Missing pose figure element: ${selector}`);
    return element;
  };
  const listeners = new AbortController(),
    signal = listeners.signal;
  const projection = query<HTMLImageElement>("[data-projection-image]");
  const deltaToggle = query<HTMLInputElement>("[data-delta-toggle]");
  const deltaCanvas = query<HTMLCanvasElement>("[data-delta-image]");
  const deltaContext = deltaCanvas.getContext("2d");
  const projectionDisplay = query<HTMLElement>("[data-projection-display]");
  const deltaScale = query<HTMLElement>("[data-delta-scale]");
  const projectionValueLabel = query<HTMLElement>(
    "[data-projection-value-label]",
  );
  const columnInput = query<HTMLInputElement>("[data-pixel-column]");
  const rowInput = query<HTMLInputElement>("[data-pixel-row]");
  const dataStatus = query<HTMLElement>("[data-data-status]");
  const sceneStatus = query<HTMLElement>("[data-scene-status]");
  const reference = data.poses.find((pose) => pose.parameter === "reference")!;
  projection.src = assetUrl(data.reference.projection);
  projection.alt = `${data.signal_label} from the synthetic pelvic CT at the reference pose.`;
  query<HTMLElement>("[data-current-pose]").textContent = poseLabel(reference);
  deltaToggle.checked = false;
  deltaToggle.disabled = true;
  projection.hidden = false;
  deltaCanvas.hidden = true;
  root.dataset.selectedPose = reference.id;
  const planeSize = data.width * data.height;
  const history = new DisplayedProjectionHistory();
  let currentPose = reference;
  let currentImage = projection;
  let referencePlanes: Float32Array | undefined;
  let finiteDifferenceStatus: Uint8Array | undefined;
  let selected = pixelIndex(
    Number(columnInput.value),
    Number(rowInput.value),
    data.width,
    data.height,
  );
  let viewer: AnatomyViewer | undefined;
  let sceneGeneration = 0;
  let nearby = false,
    disposed = false;
  let meshesPromise: Promise<ReturnType<typeof validateMeshes>> | undefined;
  let referenceFailed = false;
  const fetchBytes = async (file: string) => {
    const response = await fetch(assetUrl(file), { signal });
    if (!response.ok)
      throw new Error(`Unable to load ${file}: ${response.status}`);
    return response.arrayBuffer();
  };
  const showingDelta = () => deltaToggle.checked && !!history.current;
  const projectionSamples = () =>
    showingDelta() ? history.delta?.values : history.current?.values;
  const updateCoordinates = () => {
    const values = committedCoordinates(currentPose);
    for (const parameter of PARAMETERS)
      query<HTMLOutputElement>(`[data-pose-value="${parameter}"]`).textContent =
        `${formatValue(values[parameter])} ${parameter.startsWith("t") ? "mm" : "rad"}`;
  };
  // Presentation changes never request or commit a recorded frame.
  const updateProjectionDisplay = () => {
    const delta = showingDelta() ? history.delta : undefined;
    if (delta && deltaContext) {
      deltaCanvas.width = data.width;
      deltaCanvas.height = data.height;
      const pixels = deltaContext.createImageData(data.width, data.height);
      pixels.data.set(deltaPixels(delta));
      deltaContext.putImageData(pixels, 0, 0);
      const previous = history.previous;
      projectionDisplay.textContent = previous
        ? `Δ: ${poseLabel(currentPose)} minus ${poseLabel(previous.pose).toLowerCase()}.`
        : "Δ = 0: this is the first displayed frame, so there is no previous frame yet.";
      deltaScale.textContent = `Blue −${formatValue(delta.limit)} · neutral 0 · rust +${formatValue(delta.limit)} ${data.signal_unit}. Symmetric asinh colour scale (softening 0.02 of the range).`;
      deltaCanvas.setAttribute("aria-label", projectionDisplay.textContent);
      projectionValueLabel.textContent = "Δ projection";
      query<HTMLElement>('[data-image-channel="0"]').setAttribute(
        "aria-label",
        "Inspect the signed current-minus-previous projection difference. Arrow keys move between detector pixels.",
      );
    } else {
      projectionDisplay.textContent = data.reference.display
        ? `Logarithmic count display · 0–${formatValue(data.reference.display.projection_max)} ${data.signal_unit}`
        : "Recorded projection.";
      projectionValueLabel.textContent = "Projection";
      query<HTMLElement>('[data-image-channel="0"]').setAttribute(
        "aria-label",
        "Inspect the recorded projection pixel. Arrow keys move between detector pixels.",
      );
    }
    projection.hidden = !!delta && !!deltaContext;
    deltaCanvas.hidden = !projection.hidden;
    deltaScale.hidden = !projection.hidden;
    if (history.current)
      viewer?.setProjection(projection.hidden ? deltaCanvas : currentImage);
  };
  const updateValues = (announce = false) => {
    const column = selected % data.width,
      row = Math.floor(selected / data.width);
    root.style.setProperty(
      "--pixel-x",
      `${((column + 0.5) / data.width) * 100}%`,
    );
    root.style.setProperty(
      "--pixel-y",
      `${((row + 0.5) / data.height) * 100}%`,
    );
    columnInput.value = String(column);
    rowInput.value = String(row);
    for (const output of root.querySelectorAll<HTMLElement>(
      "[data-pixel-value]",
    )) {
      const channel = Number(output.dataset.pixelValue);
      const value =
        channel === 0
          ? projectionSamples()?.[selected]
          : referencePlanes?.[channel * planeSize + selected];
      output.textContent = value === undefined ? "—" : formatValue(value);
    }
    if (referencePlanes || history.current) {
      dataStatus.textContent = `${showingDelta() ? "Δ projection: current minus previous displayed frame" : `Projection: ${poseLabel(currentPose).toLowerCase()}`}. ${referencePlanes ? "Six derivatives: reference pose." : "Reference derivative pixel values could not load."}${history.current ? "" : " Projection pixel values are loading."}`;
      root.dataset.pixelReady = "true";
    } else if (referenceFailed)
      dataStatus.textContent =
        "Recorded pixel values could not load. The projection, six sensitivity images and their scales remain available.";
    if (finiteDifferenceStatus) {
      const unresolved = unresolvedCoordinates(
        finiteDifferenceStatus,
        selected,
      );
      query<HTMLElement>("[data-fd-status]").textContent = unresolved.length
        ? `Finite-difference check inconclusive here: ${unresolved.map((id) => parameterLabel(id).toLowerCase()).join(", ")}. These are the recorded local derivatives at the reference pose. The independent check did not resolve agreement within its numerical budget.`
        : "Finite-difference checks resolved at this reference-pose pixel.";
    }
    if (announce && (referencePlanes || history.current)) {
      const count = projectionSamples()?.[selected];
      query<HTMLElement>("[data-pixel-announcement]").textContent =
        `Column ${column}, row ${row}. ${showingDelta() ? "Delta projection" : "Projection"} ${count === undefined ? "unavailable" : `${formatValue(count)} ${data.signal_unit}`}. ${referencePlanes ? data.sensitivities.map((item, index) => `${parameterLabel(item.id)}: ${formatValue(referencePlanes![(index + 1) * planeSize + selected]!)} ${item.unit}`).join(". ") : "Reference derivative values unavailable."}`;
    }
  };
  const choosePixel = (column: number, row: number, announce = false) => {
    selected = pixelIndex(column, row, data.width, data.height);
    updateValues(announce);
  };
  for (const input of [columnInput, rowInput]) {
    input.disabled = false;
    input.addEventListener(
      "input",
      () =>
        choosePixel(Number(columnInput.value), Number(rowInput.value), true),
      { signal },
    );
  }
  for (const image of root.querySelectorAll<HTMLButtonElement>(
    "[data-image-channel]",
  )) {
    image.disabled = false;
    const point = (event: PointerEvent) => {
      const bounds = image.getBoundingClientRect();
      if (bounds.width > 0 && bounds.height > 0)
        choosePixel(
          ((event.clientX - bounds.left) / bounds.width) * data.width,
          ((event.clientY - bounds.top) / bounds.height) * data.height,
        );
    };
    image.addEventListener(
      "pointermove",
      (event) => {
        if (event.pointerType !== "touch") point(event);
      },
      { signal },
    );
    image.addEventListener("pointerdown", point, { signal });
    image.addEventListener(
      "keydown",
      (event) => {
        let column = selected % data.width,
          row = Math.floor(selected / data.width);
        if (event.key === "ArrowLeft") column--;
        else if (event.key === "ArrowRight") column++;
        else if (event.key === "ArrowUp") row--;
        else if (event.key === "ArrowDown") row++;
        else if (event.key === "Home") {
          column = 0;
          row = 0;
        } else return;
        event.preventDefault();
        choosePixel(column, row, true);
      },
      { signal },
    );
  }
  const referenceValues = fetchBytes(data.reference.channels_f32)
    .then((bytes) => {
      if (disposed) return undefined;
      referencePlanes = decodePlanes(bytes, data.width, data.height, 7);
      updateValues();
      return referencePlanes.subarray(0, planeSize);
    })
    .catch((error: unknown) => {
      if (disposed) return;
      referenceFailed = true;
      updateValues();
      console.error("Recorded sensitivity values could not load", error);
      return undefined;
    });
  // Establish the initial displayed image before a request can advance history.
  // The independent projection payload still permits interaction if the seven-
  // channel inspector payload fails to load.
  const ready = Promise.all([
    referenceValues.then(
      (values) =>
        values ??
        fetchBytes(reference.projection_f32).then((bytes) =>
          decodePlanes(bytes, data.width, data.height, 1),
        ),
    ),
    projection.decode(),
  ])
    .then(([values]) => {
      if (disposed) return false;
      history.initialise({ pose: reference, values });
      deltaToggle.disabled = !deltaContext;
      updateProjectionDisplay();
      updateValues();
      return true;
    })
    .catch((error: unknown) => {
      if (disposed) return false;
      query<HTMLElement>("[data-current-pose]").textContent =
        "Reference pose. Recorded projection data could not load. Pose changes and Δ are unavailable.";
      console.error("Initial recorded projection could not load", error);
      return false;
    });
  if (data.reference.fd_status_u8) {
    void fetchBytes(data.reference.fd_status_u8)
      .then((bytes) => {
        if (disposed) return;
        finiteDifferenceStatus = decodeFiniteDifferenceStatus(
          bytes,
          data.width,
          data.height,
        );
        updateValues();
      })
      .catch((error: unknown) => {
        if (disposed) return;
        query<HTMLElement>("[data-fd-status]").textContent =
          "The independent finite-difference check could not load. Reload the page to try again.";
        console.error("Finite-difference status could not load", error);
      });
  } else query<HTMLElement>("[data-fd-status]").hidden = true;
  const choosePose = async (requested: RecordedPose) => {
    // Resolve the record here too: neither UI coordinates nor a supplied matrix
    // may introduce an unrecorded or compounded physical pose.
    const pose = data.poses.find((candidate) => candidate.id === requested.id);
    if (!pose || disposed) return;
    const token = history.request();
    if (pose.id === currentPose.id) {
      history.fail(token); // Also cancels any older in-flight request.
      query<HTMLElement>("[data-current-pose]").textContent =
        poseLabel(currentPose);
      viewer?.setPose(currentPose);
      return;
    }
    query<HTMLElement>("[data-current-pose]").textContent =
      `${poseLabel(currentPose)}. Loading ${poseLabel(pose).toLowerCase()}…`;
    const image = new Image();
    image.src = assetUrl(pose.projection);
    try {
      const [values, , initialised] = await Promise.all([
        pose.parameter === "reference" && referencePlanes
          ? Promise.resolve(referencePlanes.subarray(0, planeSize))
          : fetchBytes(pose.projection_f32).then((bytes) =>
              decodePlanes(bytes, data.width, data.height, 1),
            ),
        image.decode(),
        ready,
      ]);
      if (disposed || !history.isLatest(token)) return;
      if (!initialised)
        throw new Error("Initial displayed projection is unavailable");
      if (!history.commit(token, { pose, values })) return;
      // The image, numeric history and mesh commit in this same event-loop turn,
      // after both required assets succeed. Pending or failed loads stay invisible.
      currentPose = pose;
      currentImage = image;
      projection.src = image.src;
      projection.alt = `${data.signal_label} from the synthetic pelvic CT. ${poseLabel(pose)}.`;
      viewer?.setPose(pose);
      updateCoordinates();
      updateProjectionDisplay();
      query<HTMLElement>("[data-current-pose]").textContent = poseLabel(pose);
      root.dataset.selectedPose = pose.id;
      updateValues();
    } catch (error) {
      if (disposed || !history.fail(token)) return;
      viewer?.setPose(currentPose);
      query<HTMLElement>("[data-current-pose]").textContent =
        `${poseLabel(currentPose)}. The requested recorded image or pixel data could not load. The displayed frame is unchanged.`;
      console.error("Recorded projection could not load", error);
    }
  };
  // These readouts also expose the exact recorded steps without WebGL. They are
  // outputs, not editable six-coordinate inputs: only one axis can be nonzero.
  for (const output of root.querySelectorAll<HTMLOutputElement>(
    "[data-pose-value]",
  )) {
    const parameter = output.dataset.poseValue as Parameter;
    output.tabIndex = 0;
    output.title =
      "Arrow keys step recorded poses for this coordinate. Home restores the reference pose.";
    output.setAttribute(
      "aria-keyshortcuts",
      "ArrowLeft ArrowRight ArrowUp ArrowDown Home",
    );
    output.setAttribute(
      "aria-label",
      `${parameterLabel(parameter)}. Displayed pose. Arrow keys select recorded steps, Home restores reference.`,
    );
    output.addEventListener(
      "keydown",
      (event) => {
        const direction =
          event.key === "ArrowRight" || event.key === "ArrowUp"
            ? 1
            : event.key === "ArrowLeft" || event.key === "ArrowDown"
              ? -1
              : 0;
        if (!direction && event.key !== "Home") return;
        event.preventDefault();
        void choosePose(
          event.key === "Home"
            ? reference
            : steppedRecordedPose(
                data,
                parameter,
                currentPose,
                direction as -1 | 1,
              ),
        );
      },
      { signal },
    );
  }
  deltaToggle.addEventListener(
    "change",
    () => {
      if (deltaToggle.disabled) deltaToggle.checked = false;
      updateProjectionDisplay();
      updateValues(true);
    },
    { signal },
  );
  const reset = query<HTMLButtonElement>("[data-reference-pose]");
  reset.disabled = false;
  reset.addEventListener(
    "click",
    () => {
      void choosePose(reference);
    },
    { signal },
  );
  updateCoordinates();
  updateProjectionDisplay();
  updateValues();

  const stopScene = () => {
    ++sceneGeneration;
    viewer?.dispose();
    viewer = undefined;
    sceneStatus.hidden = false;
    sceneStatus.textContent =
      "The 3D view resumes when the figure comes into view.";
  };
  const startScene = async () => {
    if (!nearby || document.hidden || disposed || viewer) return;
    const generation = ++sceneGeneration;
    sceneStatus.hidden = false;
    sceneStatus.textContent = "Loading pelvic bone surfaces…";
    try {
      meshesPromise ??= fetch(assetUrl("meshes.json"), { signal }).then(
        async (response) => {
          if (!response.ok)
            throw new Error(`Mesh request returned ${response.status}`);
          return validateMeshes(await response.json());
        },
      );
      const [{ createAnatomyViewer }, meshes] = await Promise.all([
        import("./viewer"),
        meshesPromise,
      ]);
      if (
        disposed ||
        generation !== sceneGeneration ||
        !nearby ||
        document.hidden
      )
        return;
      viewer = createAnatomyViewer(root, data, meshes, (pose) => {
        void choosePose(pose);
      });
      viewer.setPose(currentPose);
      if (history.current) updateProjectionDisplay();
      else
        void ready.then((initialised) => {
          if (initialised && !disposed && generation === sceneGeneration)
            updateProjectionDisplay();
        });
    } catch (error) {
      if (disposed || generation !== sceneGeneration) return;
      sceneStatus.hidden = false;
      sceneStatus.textContent =
        "The 3D view is unavailable. Recorded images and pixel inspection remain usable. Focus a pose value and use arrow keys to step recorded poses.";
      console.error("Pelvic anatomy view could not start", error);
    }
  };
  let observer: IntersectionObserver | undefined;
  if ("IntersectionObserver" in window) {
    observer = new IntersectionObserver(
      (entries) => {
        nearby = entries.some((entry) => entry.isIntersecting);
        if (nearby) void startScene();
        else stopScene();
      },
      { rootMargin: "240px" },
    );
    observer.observe(query<HTMLElement>("[data-pose-viewport]"));
  } else {
    nearby = true;
    void startScene();
  }
  document.addEventListener(
    "visibilitychange",
    () => {
      if (document.hidden) stopScene();
      else void startScene();
    },
    { signal },
  );
  return () => {
    disposed = true;
    history.dispose();
    stopScene();
    listeners.abort();
    observer?.disconnect();
    delete root.dataset.pixelReady;
  };
}

class PoseSensitivityElement extends HTMLElement {
  #observer: IntersectionObserver | undefined;
  #dispose: (() => void) | undefined;
  #generation = 0;
  connectedCallback(): void {
    const generation = ++this.#generation;
    const ready = () => {
      if (!this.isConnected || generation !== this.#generation) return;
      const start = () => {
        this.#observer?.disconnect();
        this.#observer = undefined;
        if (this.#dispose || !this.isConnected) return;
        try {
          this.#dispose = mountFigure(this);
        } catch (error) {
          console.error(
            "Pose sensitivity controls could not initialise",
            error,
          );
          const status = this.querySelector<HTMLElement>("[data-scene-status]");
          if (status)
            status.textContent =
              "Interactive controls are unavailable. The recorded images and their scales remain available.";
        }
      };
      if (!("IntersectionObserver" in window)) {
        start();
        return;
      }
      this.#observer = new IntersectionObserver(
        (entries) => {
          if (entries.some((entry) => entry.isIntersecting)) start();
        },
        { rootMargin: "240px" },
      );
      this.#observer.observe(this);
    };
    if (document.readyState === "loading")
      document.addEventListener("DOMContentLoaded", ready, { once: true });
    else queueMicrotask(ready);
  }
  disconnectedCallback(): void {
    ++this.#generation;
    this.#observer?.disconnect();
    this.#observer = undefined;
    this.#dispose?.();
    this.#dispose = undefined;
  }
}

export function mountPoseSensitivityElements(): void {
  if (!customElements.get(ELEMENT))
    customElements.define(ELEMENT, PoseSensitivityElement);
}
