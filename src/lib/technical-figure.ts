const ELEMENT_NAME = "dpt-technical-figure";

function supportsWebGL(): boolean {
  try {
    const probe = document.createElement("canvas");
    const context = probe.getContext("webgl2") ?? probe.getContext("webgl");
    if (!context) return false;
    context.getExtension("WEBGL_lose_context")?.loseContext();
    return true;
  } catch {
    return false;
  }
}

class TechnicalFigureElement extends HTMLElement {
  #intersectionObserver: IntersectionObserver | undefined;
  #resizeObserver: ResizeObserver | undefined;
  #teardown: (() => void) | undefined;
  #started = false;
  #generation = 0;

  connectedCallback(): void {
    if (!supportsWebGL()) {
      this.#setStatus(
        "Interactive rendering is unavailable. Showing the static diagram.",
      );
      return;
    }

    if (!("IntersectionObserver" in window)) {
      void this.#start();
      return;
    }

    this.#intersectionObserver = new IntersectionObserver(
      (entries) => {
        if (entries.some((entry) => entry.isIntersecting)) {
          this.#intersectionObserver?.disconnect();
          this.#intersectionObserver = undefined;
          void this.#start();
        }
      },
      { rootMargin: "240px" },
    );
    this.#intersectionObserver.observe(this);
  }

  disconnectedCallback(): void {
    this.#generation += 1;
    this.#intersectionObserver?.disconnect();
    this.#resizeObserver?.disconnect();
    this.#teardown?.();
    this.#intersectionObserver = undefined;
    this.#resizeObserver = undefined;
    this.#teardown = undefined;
    this.#started = false;
    this.#showFallback();
  }

  async #start(): Promise<void> {
    if (this.#started || !this.isConnected) return;
    this.#started = true;
    const generation = this.#generation;

    const canvas = this.querySelector("canvas");
    const fallback = this.querySelector<SVGElement>(
      ".interactive-figure__fallback",
    );
    const frame = this.querySelector<HTMLElement>(".interactive-figure__frame");
    if (!(canvas instanceof HTMLCanvasElement) || !frame) return;

    try {
      const THREE = await import("three");
      if (!this.isConnected || generation !== this.#generation) return;

      const renderer = new THREE.WebGLRenderer({
        canvas,
        antialias: true,
        alpha: true,
      });
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
      renderer.setClearColor(0xf1eee7, 1);
      const listeners = new AbortController();
      const disposables: { dispose(): void }[] = [];
      this.#teardown = () => {
        listeners.abort();
        for (const resource of disposables) resource.dispose();
        renderer.dispose();
        renderer.forceContextLoss();
      };

      const scene = new THREE.Scene();
      const camera = new THREE.PerspectiveCamera(32, 1, 0.1, 100);
      camera.position.set(0.2, 2.7, 7.8);
      camera.lookAt(0.25, 0, 0);

      const sourceGeometry = new THREE.SphereGeometry(0.14, 20, 12);
      const sourceMaterial = new THREE.MeshBasicMaterial({ color: 0xa4442d });
      const source = new THREE.Mesh(sourceGeometry, sourceMaterial);
      source.position.x = -2.55;

      const rayGeometry = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(-2.4, 0, 0),
        new THREE.Vector3(2.25, 0, 0),
      ]);
      const rayMaterial = new THREE.LineBasicMaterial({ color: 0x856e3f });
      const ray = new THREE.Line(rayGeometry, rayMaterial);

      const detectorGeometry = new THREE.PlaneGeometry(2.5, 3.1, 4, 5);
      const detectorMaterial = new THREE.MeshBasicMaterial({
        color: 0x344a53,
        side: THREE.DoubleSide,
        wireframe: true,
      });
      const detector = new THREE.Mesh(detectorGeometry, detectorMaterial);
      detector.position.x = 2.45;
      detector.rotation.y = Math.PI / 2;

      scene.add(source, ray, detector);
      disposables.push(
        sourceGeometry,
        sourceMaterial,
        rayGeometry,
        rayMaterial,
        detectorGeometry,
        detectorMaterial,
      );

      const initialYaw = Math.atan2(-0.05, 7.8);
      const initialPitch = Math.atan2(2.7, Math.hypot(-0.05, 7.8));
      let yaw = initialYaw;
      let pitch = initialPitch;
      let zoom = 1;
      let pointer: { id: number; x: number; y: number } | undefined;

      const paint = (): void => {
        // Fit the same schematic geometry on narrow screens; this is camera
        // navigation only, not a transport calculation or physical parameter.
        const distance =
          Math.max(
            8.3,
            3.1 / (Math.tan((16 * Math.PI) / 180) * camera.aspect),
          ) * zoom;
        camera.position.set(
          0.25 + distance * Math.cos(pitch) * Math.sin(yaw),
          distance * Math.sin(pitch),
          distance * Math.cos(pitch) * Math.cos(yaw),
        );
        camera.lookAt(0.25, 0, 0);
        renderer.render(scene, camera);
        this.dataset.viewYaw = yaw.toFixed(3);
        this.dataset.viewZoom = zoom.toFixed(3);
      };

      const render = (): void => {
        const width = Math.max(frame.clientWidth, 1);
        const height = Math.max(frame.clientHeight, 1);
        renderer.setSize(width, height, false);
        camera.aspect = width / height;
        camera.updateProjectionMatrix();
        paint();
      };

      const rotate = (horizontal: number, vertical: number): void => {
        yaw += horizontal;
        pitch = Math.max(-1.35, Math.min(1.35, pitch + vertical));
        paint();
      };
      const changeZoom = (factor: number): void => {
        zoom = Math.max(0.65, Math.min(1.8, zoom * factor));
        paint();
      };
      const reset = (): void => {
        yaw = initialYaw;
        pitch = initialPitch;
        zoom = 1;
        paint();
      };
      const releasePointer = (): void => {
        if (pointer && canvas.hasPointerCapture(pointer.id))
          canvas.releasePointerCapture(pointer.id);
        pointer = undefined;
        delete this.dataset.dragging;
      };
      canvas.addEventListener(
        "pointerdown",
        (event) => {
          if (!event.isPrimary || event.button !== 0) return;
          pointer = { id: event.pointerId, x: event.clientX, y: event.clientY };
          canvas.setPointerCapture(event.pointerId);
          canvas.focus({ preventScroll: true });
          this.dataset.dragging = "true";
        },
        { signal: listeners.signal },
      );
      canvas.addEventListener(
        "pointermove",
        (event) => {
          if (!pointer || event.pointerId !== pointer.id) return;
          rotate(
            (pointer.x - event.clientX) * 0.008,
            (event.clientY - pointer.y) * 0.008,
          );
          pointer.x = event.clientX;
          pointer.y = event.clientY;
        },
        { signal: listeners.signal },
      );
      for (const event of [
        "pointerup",
        "pointercancel",
        "lostpointercapture",
      ] as const) {
        canvas.addEventListener(event, releasePointer, {
          signal: listeners.signal,
        });
      }
      canvas.addEventListener(
        "keydown",
        (event) => {
          const step = event.shiftKey ? 0.2 : 0.08;
          if (event.key === "ArrowLeft") rotate(-step, 0);
          else if (event.key === "ArrowRight") rotate(step, 0);
          else if (event.key === "ArrowUp") rotate(0, step);
          else if (event.key === "ArrowDown") rotate(0, -step);
          else if (event.key === "+" || event.key === "=") changeZoom(0.9);
          else if (event.key === "-") changeZoom(1 / 0.9);
          else if (event.key === "Home") reset();
          else return;
          event.preventDefault();
        },
        { signal: listeners.signal },
      );
      for (const button of this.querySelectorAll<HTMLButtonElement>(
        "[data-figure-action]",
      )) {
        button.disabled = false;
        button.addEventListener(
          "click",
          () => {
            if (button.dataset.figureAction === "zoom-in") changeZoom(0.9);
            else if (button.dataset.figureAction === "zoom-out")
              changeZoom(1 / 0.9);
            else reset();
          },
          { signal: listeners.signal },
        );
      }
      canvas.addEventListener(
        "webglcontextlost",
        (event) => {
          event.preventDefault();
          this.#resizeObserver?.disconnect();
          this.#teardown?.();
          this.#teardown = undefined;
          this.#showFallback();
          this.#setStatus(
            "Interactive rendering was interrupted. Showing the static diagram.",
            true,
          );
        },
        { signal: listeners.signal },
      );

      this.#resizeObserver = new ResizeObserver(render);
      this.#resizeObserver.observe(frame);
      render();

      canvas.hidden = false;
      canvas.tabIndex = 0;
      if (fallback) fallback.setAttribute("hidden", "");
      const controls = this.querySelector<HTMLElement>(
        "[data-figure-controls]",
      );
      if (controls) controls.hidden = false;
      this.dataset.interactiveReady = "true";
      this.#setStatus("Drag to explore the geometry");
    } catch (error) {
      this.#started = false;
      this.#resizeObserver?.disconnect();
      this.#teardown?.();
      this.#teardown = undefined;
      this.#showFallback();
      this.#setStatus(
        "Interactive rendering failed. Showing the static diagram.",
        true,
      );
      console.error("Unable to initialise the technical figure.", error);
    }
  }

  #showFallback(): void {
    const canvas = this.querySelector("canvas");
    if (canvas) {
      canvas.hidden = true;
      canvas.tabIndex = -1;
    }
    this.querySelector(".interactive-figure__fallback")?.removeAttribute(
      "hidden",
    );
    const controls = this.querySelector<HTMLElement>("[data-figure-controls]");
    if (controls) controls.hidden = true;
    for (const button of this.querySelectorAll<HTMLButtonElement>(
      "[data-figure-action]",
    ))
      button.disabled = true;
    delete this.dataset.interactiveReady;
    delete this.dataset.dragging;
  }

  #setStatus(message: string, error = false): void {
    const status = this.querySelector<HTMLElement>("[data-figure-status]");
    if (status) {
      status.textContent = message;
      status.dataset.figureError = String(error);
    }
  }
}

export function mountTechnicalFigureElements(): void {
  if (!customElements.get(ELEMENT_NAME)) {
    customElements.define(ELEMENT_NAME, TechnicalFigureElement);
  }
}
