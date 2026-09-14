import type * as Three from "three";

const ELEMENT = "dpt-anatomy-figure";
type Pose = "standing" | "supine" | "prone";

class AnatomyFigureElement extends HTMLElement {
  #observer?: IntersectionObserver;
  #dispose: (() => void) | undefined;
  #generation = 0;

  connectedCallback(): void {
    const generation = ++this.#generation;
    const start = (): void => {
      this.#observer?.disconnect();
      void this.#mount(generation);
    };
    if (!("IntersectionObserver" in window)) start();
    else {
      this.#observer = new IntersectionObserver(
        (entries) => {
          if (entries.some((entry) => entry.isIntersecting)) start();
        },
        { rootMargin: "240px" },
      );
      this.#observer.observe(this);
    }
  }

  disconnectedCallback(): void {
    ++this.#generation;
    this.#observer?.disconnect();
    this.#dispose?.();
    this.#dispose = undefined;
    this.#fallback("Static anatomical diagram.");
  }

  #fallback(message: string): void {
    const canvas = this.querySelector("canvas");
    if (canvas) {
      canvas.hidden = true;
      canvas.tabIndex = -1;
    }
    this.querySelector(".anatomy-fallback")?.removeAttribute("hidden");
    const labels = this.querySelector<HTMLElement>(".anatomy-labels");
    if (labels) labels.hidden = true;
    for (const button of this.querySelectorAll<HTMLButtonElement>(
      "[data-anatomy-pose], [data-anatomy-action]",
    ))
      button.disabled = true;
    const status = this.querySelector("[data-anatomy-status]");
    if (status) status.textContent = message;
    delete this.dataset.interactiveReady;
  }

  async #mount(generation: number): Promise<void> {
    const canvas = this.querySelector<HTMLCanvasElement>(
      ".anatomy-frame canvas",
    );
    const frame = this.querySelector<HTMLElement>(".anatomy-frame");
    const layer = this.querySelector<HTMLElement>(".anatomy-labels");
    if (!canvas || !frame || !layer) return;
    const resources = new Set<{ dispose(): void }>();
    const listeners = new AbortController();
    let renderer: Three.WebGLRenderer | undefined;
    let resize: ResizeObserver | undefined;
    let controls:
      | import("three/examples/jsm/controls/OrbitControls.js").OrbitControls
      | undefined;
    let stopped = false;
    const dispose = (): void => {
      if (stopped) return;
      stopped = true;
      listeners.abort();
      resize?.disconnect();
      controls?.dispose();
      for (const resource of resources) resource.dispose();
      renderer?.dispose();
      renderer?.forceContextLoss();
      layer.replaceChildren();
    };
    this.#dispose = dispose;
    try {
      const [T, { OrbitControls }, { GLTFLoader }] = await Promise.all([
        import("three"),
        import("three/examples/jsm/controls/OrbitControls.js"),
        import("three/examples/jsm/loaders/GLTFLoader.js"),
      ]);
      if (stopped || generation !== this.#generation) return;
      renderer = new T.WebGLRenderer({ canvas, antialias: true, alpha: true });
      renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
      renderer.setClearColor(0xf2efe8, 1);
      renderer.autoClear = false;
      const scene = new T.Scene();
      const body = new T.Group();
      scene.add(body);
      const camera = new T.PerspectiveCamera(34, 1, 0.05, 100);
      camera.position.set(3.1, 1.5, 7);
      controls = new OrbitControls(camera, canvas);
      canvas.style.touchAction = "pan-y";
      controls.enableDamping = false;
      controls.enablePan = false;
      controls.enableZoom = true;
      controls.target.set(0, 0.35, 0);
      controls.minDistance = 2.8;
      controls.maxDistance = 18;
      const keep = <V extends { dispose(): void }>(resource: V): V => {
        resources.add(resource);
        return resource;
      };
      const material = keep(
        new T.MeshBasicMaterial({
          color: 0xece8df,
          side: T.DoubleSide,
          polygonOffset: true,
          polygonOffsetFactor: 1,
          polygonOffsetUnits: 1,
        }),
      );
      const lineMaterial = keep(
        new T.LineBasicMaterial({
          color: 0x52646a,
          transparent: true,
          opacity: 0.78,
        }),
      );
      type Silhouette = {
        mesh: Three.Mesh;
        edges: {
          a: Three.Vector3;
          b: Three.Vector3;
          normals: Three.Vector3[];
        }[];
        geometry: Three.BufferGeometry;
        positions: Float32Array;
      };
      const silhouettes: Silhouette[] = [];
      const gltf = await new GLTFLoader().loadAsync(
        this.dataset.model ?? "/assets/anatomy/body.glb",
      );
      // Register the loader-owned resources even if disconnect happened during fetch.
      gltf.scene.traverse((object) => {
        if (!(object instanceof T.Mesh)) return;
        resources.add(object.geometry);
        for (const old of Array.isArray(object.material)
          ? object.material
          : [object.material]) {
          for (const value of Object.values(old))
            if (value instanceof T.Texture) resources.add(value);
          resources.add(old);
        }
      });
      if (stopped || generation !== this.#generation) {
        for (const resource of resources) resource.dispose();
        return;
      }
      body.add(gltf.scene);
      gltf.scene.traverse((object) => {
        if (!(object instanceof T.Mesh)) return;
        object.material = material;
        const positions = object.geometry.getAttribute("position");
        const index = object.geometry.index;
        const edgeMap = new Map<string, Silhouette["edges"][number]>();
        const point = (i: number): Three.Vector3 =>
          new T.Vector3().fromBufferAttribute(
            positions,
            index ? index.getX(i) : i,
          );
        const key = (v: Three.Vector3): string =>
          `${Math.round(v.x * 1e6)},${Math.round(v.y * 1e6)},${Math.round(v.z * 1e6)}`;
        for (let i = 0; i < (index?.count ?? positions.count); i += 3) {
          const vertices = [point(i), point(i + 1), point(i + 2)] as const;
          const normal = vertices[1]
            .clone()
            .sub(vertices[0])
            .cross(vertices[2].clone().sub(vertices[0]))
            .normalize();
          if (normal.lengthSq() === 0) continue;
          for (let j = 0; j < 3; j++) {
            const a = vertices[j]!,
              b = vertices[(j + 1) % 3]!;
            const ka = key(a),
              kb = key(b),
              id = ka < kb ? `${ka}|${kb}` : `${kb}|${ka}`;
            const edge = edgeMap.get(id);
            if (edge) edge.normals.push(normal);
            else edgeMap.set(id, { a, b, normals: [normal] });
          }
        }
        const edges = [...edgeMap.values()];
        const output = new Float32Array(edges.length * 6);
        const geometry = keep(new T.BufferGeometry());
        geometry.setAttribute(
          "position",
          new T.BufferAttribute(output, 3).setUsage(T.DynamicDrawUsage),
        );
        const lines = new T.LineSegments(geometry, lineMaterial);
        lines.frustumCulled = false;
        object.add(lines);
        silhouettes.push({ mesh: object, edges, geometry, positions: output });
      });

      const labels: { element: HTMLSpanElement; anchor: Three.Object3D }[] = [];
      const arrow = (
        start: Three.Vector3,
        direction: Three.Vector3,
        length: number,
        text: string,
        kind = "axis",
        colour = 0x9b4330,
      ): void => {
        const helper = new T.ArrowHelper(
          direction.clone().normalize(),
          start,
          length,
          colour,
          0.075,
          0.035,
        );
        body.add(helper);
        helper.traverse((object) => {
          if (object instanceof T.Mesh || object instanceof T.Line) {
            resources.add(object.geometry);
            for (const m of Array.isArray(object.material)
              ? object.material
              : [object.material])
              resources.add(m);
          }
        });
        const anchor = new T.Object3D();
        anchor.position
          .copy(start)
          .addScaledVector(direction.clone().normalize(), length + 0.1);
        body.add(anchor);
        const element = document.createElement("span");
        element.className = "anatomy-label";
        element.dataset.kind = kind;
        element.textContent = text;
        element.style.position = "absolute";
        element.style.whiteSpace = "nowrap";
        element.style.transform = "translate(-50%, -50%)";
        layer.append(element);
        labels.push({ element, anchor });
      };
      // Offset from the midline so the longitudinal direction remains legible.
      const spinalAxis = new T.Vector3(-0.48, 0.1, -0.24);
      arrow(
        spinalAxis,
        new T.Vector3(0, 1, 0),
        0.63,
        "Cephalad",
        "axis",
        0x506771,
      );
      arrow(
        spinalAxis,
        new T.Vector3(0, -1, 0),
        0.63,
        "Caudal",
        "axis",
        0x506771,
      );

      const planes = new T.Group();
      planes.visible = false;
      body.add(planes);
      for (const [name, colour, rotation] of [
        ["Sagittal", 0x0072b2, [0, Math.PI / 2, 0]],
        ["Coronal", 0xe69f00, [0, 0, 0]],
        ["Transverse", 0x009e73, [-Math.PI / 2, 0, 0]],
      ] as const) {
        const mesh = new T.Mesh(
          keep(new T.PlaneGeometry(1.8, name === "Transverse" ? 1.3 : 2.2)),
          keep(
            new T.MeshBasicMaterial({
              color: colour,
              transparent: true,
              opacity: 0.13,
              side: T.DoubleSide,
              depthWrite: false,
            }),
          ),
        );
        mesh.rotation.set(rotation[0], rotation[1], rotation[2]);
        planes.add(mesh);
        // Text is actual plane-local geometry, not a projected HTML overlay.
        const tile = document.createElement("canvas");
        tile.width = 512;
        tile.height = 96;
        const context = tile.getContext("2d");
        if (!context) throw new Error("Unable to create plane labels.");
        context.font = "48px sans-serif";
        context.fillStyle = "#344a53";
        context.textAlign = "center";
        context.textBaseline = "middle";
        context.fillText(name, 256, 48);
        const texture = keep(new T.CanvasTexture(tile));
        texture.colorSpace = T.SRGBColorSpace;
        const textGeometry = keep(new T.PlaneGeometry(0.8, 0.15));
        const textMaterial = keep(
          new T.MeshBasicMaterial({
            map: texture,
            transparent: true,
            depthWrite: false,
            side: T.FrontSide,
          }),
        );
        // Separate front/back faces keep the lettering readable on either side.
        for (const side of [1, -1]) {
          const text = new T.Mesh(textGeometry, textMaterial);
          text.position.set(
            0.44,
            (name === "Transverse" ? 0.65 : 1.1) - 0.12,
            side * 0.003,
          );
          text.rotation.y = side === 1 ? 0 : Math.PI;
          text.renderOrder = 1;
          mesh.add(text);
        }
      }
      const hub = new T.Vector3(0, 1.6, 0);
      for (const [direction, label] of [
        [[0, 1, 0], "Superior"],
        [[0, -1, 0], "Inferior"],
        [[0, 0, 1], "Anterior"],
        [[0, 0, -1], "Posterior"],
        [[1, 0, 0], "Left"],
        [[-1, 0, 0], "Right"],
      ] as const)
        arrow(hub, new T.Vector3(...direction), 0.38, label);
      const support = new T.Mesh(
        keep(new T.PlaneGeometry(1.4, 2.7)),
        keep(
          new T.MeshBasicMaterial({
            color: 0xe1dcd1,
            transparent: true,
            opacity: 0.65,
            side: T.DoubleSide,
          }),
        ),
      );
      support.rotation.x = -Math.PI / 2;
      support.position.y = -0.25;
      support.visible = false;
      scene.add(support);

      const cubeScene = new T.Scene();
      const cubeCamera = new T.PerspectiveCamera(32, 1, 0.1, 20);
      cubeCamera.position.z = 4.5;
      const cube = new T.Mesh(
        keep(new T.BoxGeometry(1, 1, 1)),
        ["L", "R", "S", "I", "A", "P"].map((letter) => {
          const tile = document.createElement("canvas");
          tile.width = tile.height = 128;
          const context = tile.getContext("2d");
          if (!context) throw new Error("Unable to create orientation labels.");
          context.fillStyle = "#eeeae1";
          context.fillRect(0, 0, 128, 128);
          context.strokeStyle = "#a7a295";
          context.strokeRect(1, 1, 126, 126);
          context.fillStyle = "#344a53";
          context.font = "54px serif";
          context.textAlign = "center";
          context.textBaseline = "middle";
          context.fillText(letter, 64, 67);
          const texture = keep(new T.CanvasTexture(tile));
          texture.colorSpace = T.SRGBColorSpace;
          return keep(new T.MeshBasicMaterial({ map: texture }));
        }),
      );
      cubeScene.add(cube);
      let pose: Pose = "standing";
      const localCamera = new T.Vector3();
      const view = new T.Vector3();
      const projected = new T.Vector3();
      const labelWorld = new T.Vector3();
      const labelDirection = new T.Vector3();
      const labelRay = new T.Raycaster();
      const labelHits: Three.Intersection[] = [];
      // Only the opaque body occludes annotations, not transparent guides or arrows.
      const labelOccluders = silhouettes.map(({ mesh }) => mesh);
      const orientation = new T.Quaternion();
      const render = (): void => {
        if (stopped || !renderer) return;
        const width = Math.max(frame.clientWidth, 1),
          height = Math.max(frame.clientHeight, 1);
        scene.updateMatrixWorld(true);
        for (const contour of silhouettes) {
          localCamera.copy(camera.position);
          contour.mesh.worldToLocal(localCamera);
          let count = 0;
          for (const edge of contour.edges) {
            view.copy(localCamera).sub(edge.a);
            const front = edge.normals.some((n) => n.dot(view) > 0);
            const back = edge.normals.some((n) => n.dot(view) <= 0);
            // A contour is an adjacent front/back transition, not a wireframe.
            if (!(front && (back || edge.normals.length === 1))) continue;
            edge.a.toArray(contour.positions, count);
            count += 3;
            edge.b.toArray(contour.positions, count);
            count += 3;
          }
          contour.geometry.setDrawRange(0, count / 3);
          contour.geometry.getAttribute("position").needsUpdate = true;
        }
        renderer.setViewport(0, 0, width, height);
        renderer.setScissorTest(false);
        renderer.clear();
        renderer.render(scene, camera);
        const placed: { x: number; y: number; w: number }[] = [];
        for (const { element, anchor } of labels) {
          anchor.getWorldPosition(labelWorld);
          projected.copy(labelWorld).project(camera);
          labelRay.set(
            camera.position,
            labelDirection.copy(labelWorld).sub(camera.position).normalize(),
          );
          // Stop just before the anchor to avoid self-occlusion at a surface.
          labelRay.far = Math.max(
            0,
            camera.position.distanceTo(labelWorld) - 0.005,
          );
          labelHits.length = 0;
          labelRay.intersectObjects(labelOccluders, false, labelHits);
          if (projected.z < -1 || projected.z > 1 || labelHits.length > 0) {
            element.hidden = true;
            continue;
          }
          element.hidden = false;
          const w = element.offsetWidth || 64;
          const x = Math.min(
            width - w / 2 - 8,
            Math.max(w / 2 + 8, ((projected.x + 1) * width) / 2),
          );
          let y = Math.min(
            height - 18,
            Math.max(18, ((-projected.y + 1) * height) / 2),
          );
          for (
            let attempt = 0;
            attempt < 5 &&
            placed.some(
              (p) =>
                Math.abs(p.x - x) < (p.w + w) / 2 + 4 && Math.abs(p.y - y) < 20,
            );
            attempt++
          )
            y = Math.min(height - 18, y + 21);
          element.style.left = `${x}px`;
          element.style.top = `${y}px`;
          placed.push({ x, y, w });
        }
        cube.quaternion
          .copy(camera.getWorldQuaternion(orientation).invert())
          .multiply(body.quaternion);
        const size = Math.min(110, width * 0.25);
        renderer.clearDepth();
        renderer.setViewport(width - size - 10, height - size - 10, size, size);
        renderer.render(cubeScene, cubeCamera);
        renderer.setViewport(0, 0, width, height);
        this.dataset.pose = pose;
      };
      const resizeFrame = (): void => {
        if (!renderer || !controls) return;
        const width = Math.max(frame.clientWidth, 1),
          height = Math.max(frame.clientHeight, 1);
        renderer.setSize(width, height, false);
        camera.aspect = width / height;
        camera.updateProjectionMatrix();
        render();
      };
      const fit = (): void => {
        if (!controls) return;
        resizeFrame();
        const distance = Math.max(5.8, 2.8 / camera.aspect);
        camera.position
          .set(0.3, pose === "standing" ? 0.18 : 1.1, 1)
          .normalize()
          .multiplyScalar(distance)
          .add(controls.target);
        controls.update();
        render();
      };
      const setPose = (next: Pose): void => {
        pose = next;
        body.rotation.x =
          next === "supine" ? -Math.PI / 2 : next === "prone" ? Math.PI / 2 : 0;
        support.visible = next !== "standing";
        body.updateMatrixWorld(true);
        support.position.y =
          new T.Box3().setFromObject(gltf.scene).min.y - 0.025;
        controls!.target.set(0, next === "standing" ? 0.45 : 0, 0);
        for (const button of this.querySelectorAll<HTMLButtonElement>(
          "[data-anatomy-pose]",
        ))
          button.setAttribute(
            "aria-pressed",
            String(button.dataset.anatomyPose === next),
          );
        fit();
        const status = this.querySelector("[data-anatomy-status]");
        if (status)
          status.textContent = `${next === "standing" ? "Standing" : next === "supine" ? "Supine: anterior faces upwards" : "Prone: anterior faces downwards"}. Anatomical directions remain attached to the body.`;
      };
      const zoom = (factor: number): void => {
        if (!controls) return;
        const offset = camera.position.clone().sub(controls.target);
        offset.setLength(
          Math.max(
            controls.minDistance,
            Math.min(controls.maxDistance, offset.length() * factor),
          ),
        );
        camera.position.copy(controls.target).add(offset);
        controls.update();
        render();
      };
      controls.addEventListener("change", render);
      canvas.addEventListener(
        "keydown",
        (event) => {
          if (!controls) return;
          if (event.key === "Home") setPose("standing");
          else if (event.key === "+" || event.key === "=") zoom(0.9);
          else if (event.key === "-") zoom(1 / 0.9);
          else if (event.key.startsWith("Arrow")) {
            const offset = camera.position.clone().sub(controls.target);
            const spherical = new T.Spherical().setFromVector3(offset);
            if (event.key === "ArrowLeft") spherical.theta -= 0.12;
            if (event.key === "ArrowRight") spherical.theta += 0.12;
            if (event.key === "ArrowUp") spherical.phi -= 0.12;
            if (event.key === "ArrowDown") spherical.phi += 0.12;
            spherical.makeSafe();
            camera.position
              .copy(controls.target)
              .add(offset.setFromSpherical(spherical));
            controls.update();
            render();
          } else return;
          event.preventDefault();
        },
        { signal: listeners.signal },
      );
      for (const button of this.querySelectorAll<HTMLButtonElement>(
        "[data-anatomy-pose], [data-anatomy-action]",
      )) {
        button.disabled = false;
        if (button.dataset.anatomyAction === "planes") {
          button.setAttribute("aria-pressed", "false");
          button.setAttribute("aria-label", "Show anatomical planes");
          button.title = "Show anatomical planes";
        }
        button.addEventListener(
          "click",
          () => {
            const next = button.dataset.anatomyPose;
            if (next === "standing" || next === "supine" || next === "prone")
              setPose(next);
            else if (button.dataset.anatomyAction === "zoom-in") zoom(0.9);
            else if (button.dataset.anatomyAction === "zoom-out") zoom(1 / 0.9);
            else if (button.dataset.anatomyAction === "planes") {
              planes.visible = !planes.visible;
              button.setAttribute("aria-pressed", String(planes.visible));
              const label = `${planes.visible ? "Hide" : "Show"} anatomical planes`;
              button.setAttribute("aria-label", label);
              button.title = label;
              render();
            } else setPose("standing");
          },
          { signal: listeners.signal },
        );
      }
      canvas.addEventListener(
        "webglcontextlost",
        (event) => {
          event.preventDefault();
          dispose();
          this.#fallback(
            "Interactive rendering was interrupted. Showing the static anatomy diagram.",
          );
        },
        { signal: listeners.signal },
      );
      resize = new ResizeObserver(resizeFrame);
      resize.observe(frame);
      canvas.hidden = false;
      canvas.tabIndex = 0;
      layer.hidden = false;
      this.querySelector(".anatomy-fallback")?.setAttribute("hidden", "");
      this.dataset.interactiveReady = "true";
      setPose("standing");
    } catch (error) {
      dispose();
      if (generation === this.#generation)
        this.#fallback(
          "Interactive anatomy is unavailable. Showing the static diagram.",
        );
      console.warn("Unable to initialise anatomical figure.", error);
    }
  }
}

export function mountAnatomyFigureElements(): void {
  if (!customElements.get(ELEMENT))
    customElements.define(ELEMENT, AnatomyFigureElement);
}
