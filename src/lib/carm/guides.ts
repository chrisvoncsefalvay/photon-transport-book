import * as T from "three";
import { labelsByMode } from "./kinematics";
import type { JointSettings, TerminologyMode } from "./kinematics";

export type ReferenceFrame = "world" | "patient" | "detector";
export interface GuideState {
  isoMatrix: T.Matrix4;
  referenceMatrix: T.Matrix4;
  reference: ReferenceFrame;
  hover: keyof JointSettings | null;
  mode: TerminologyMode;
  translationAnchor?: T.Vector3;
  angulationMatrix: T.Matrix4;
  orbitalMatrix: T.Matrix4;
}

/** Chassis/lift travel stays in world coordinates as the arm rotates. */
export function translationDirection(
  key: "insertionMm" | "liftMm" | "longitudinalMm",
  sign: 1 | -1,
) {
  if (key === "insertionMm") return new T.Vector3(sign, 0, 0);
  if (key === "liftMm") return new T.Vector3(0, sign, 0);
  return new T.Vector3(0, 0, sign);
}

/** Signed orthogonal detector-plane coordinates in scene units, not pixels. */
export function detectorUV(
  pointWorld: T.Vector3,
  detectorWorldMatrix: T.Matrix4,
) {
  const local = pointWorld
    .clone()
    .applyMatrix4(detectorWorldMatrix.clone().invert());
  return { u: local.x, v: -local.z, normalDistance: local.y };
}

/** Picture plane through the isocentre, perpendicular to the central ray. */
export function imagingPlane(isoMatrix: T.Matrix4): T.Plane {
  const origin = new T.Vector3().setFromMatrixPosition(isoMatrix);
  const normal = new T.Vector3(0, 1, 0).transformDirection(isoMatrix);
  return new T.Plane().setFromNormalAndCoplanarPoint(normal, origin);
}

/** Detector axes are U=X, V=-Z, N=Y, so U cross V = N. */
export function referenceBasis(matrix: T.Matrix4, reference: ReferenceFrame) {
  return {
    origin: new T.Vector3().setFromMatrixPosition(matrix),
    x: new T.Vector3(1, 0, 0).transformDirection(matrix),
    y: new T.Vector3(
      0,
      reference === "detector" ? 0 : 1,
      reference === "detector" ? -1 : 0,
    ).transformDirection(matrix),
    z: new T.Vector3(
      0,
      reference === "detector" ? 1 : 0,
      reference === "detector" ? 0 : 1,
    ).transformDirection(matrix),
  };
}

/** Positive joint travel rotates by a negative right-handed angle. */
export function rotationArc(
  axis: T.Vector3,
  radial: T.Vector3,
  sign: 1 | -1,
  radius = 0.32,
  angle = 0.72,
) {
  const unitAxis = axis.clone().normalize();
  const initial = radial
    .clone()
    .addScaledVector(unitAxis, -radial.dot(unitAxis))
    .normalize();
  const points: T.Vector3[] = [];
  for (let i = 0; i <= 24; i++)
    points.push(
      initial
        .clone()
        .multiplyScalar(radius)
        .applyAxisAngle(unitAxis, (-sign * angle * i) / 24),
    );
  const end = points[24]!;
  const tangent = new T.Vector3()
    .crossVectors(unitAxis, end)
    .multiplyScalar(-sign)
    .normalize();
  return { points, end: end.clone(), tangent };
}

/** Visual guides only: these objects never modify the articulated machine. */
export function createGuides(scene: T.Scene) {
  const root = new T.Group();
  root.name = "c-arm-visual-guides";
  scene.add(root);
  const resources = new Set<{ dispose(): void }>();
  const keep = <V extends { dispose(): void }>(resource: V): V => {
    resources.add(resource);
    return resource;
  };
  const register = (object: T.Object3D) =>
    object.traverse((child) => {
      if (child instanceof T.Mesh || child instanceof T.Line) {
        keep(child.geometry);
        for (const material of Array.isArray(child.material)
          ? child.material
          : [child.material])
          keep(material);
      }
    });
  const arrow = (
    direction: T.Vector3,
    origin: T.Vector3,
    length: number,
    colour: number,
  ) => {
    const object = new T.ArrowHelper(
      direction,
      origin,
      length,
      colour,
      0.035,
      0.021,
    );
    object.traverse((child) => {
      child.renderOrder = 12;
      if (child instanceof T.Mesh || child instanceof T.Line) {
        for (const material of Array.isArray(child.material)
          ? child.material
          : [child.material]) {
          material.depthTest = false;
          material.depthWrite = false;
          material.transparent = true;
          material.opacity = 0.92;
        }
      }
    });
    register(object);
    return object;
  };
  const textSprites: { sprite: T.Sprite; setText(text: string): void }[] = [];
  const label = (text: string, position: T.Vector3, height = 0.035) => {
    const canvas = document.createElement("canvas");
    canvas.width = 768;
    canvas.height = 80;
    const context = canvas.getContext("2d")!;
    const texture = keep(new T.CanvasTexture(canvas));
    texture.colorSpace = T.SRGBColorSpace;
    const material = keep(
      new T.SpriteMaterial({
        map: texture,
        transparent: true,
        depthTest: false,
        depthWrite: false,
      }),
    );
    const sprite = new T.Sprite(material);
    sprite.position.copy(position);
    sprite.scale.set((height * canvas.width) / canvas.height, height, 1);
    sprite.renderOrder = 14;
    const setText = (value: string) => {
      context.clearRect(0, 0, canvas.width, canvas.height);
      context.font = "32px Arial";
      context.textAlign = "center";
      context.textBaseline = "middle";
      const width = context.measureText(value).width + 18;
      context.fillStyle = "rgba(249,247,240,.9)";
      context.fillRect((canvas.width - width) / 2, 13, width, 54);
      context.fillStyle = "#344a53";
      context.fillText(value, canvas.width / 2, canvas.height / 2);
      texture.needsUpdate = true;
    };
    setText(text);
    const entry = { sprite, setText };
    textSprites.push(entry);
    return entry;
  };
  const plane = new T.Group();
  plane.name = "imaging-plane";
  plane.matrixAutoUpdate = false;
  root.add(plane);
  const surface = new T.Mesh(
    keep(new T.PlaneGeometry(1.06, 0.92).rotateX(-Math.PI / 2)),
    keep(
      new T.MeshBasicMaterial({
        color: 0x657f99,
        transparent: true,
        opacity: 0.08,
        side: T.DoubleSide,
        depthWrite: false,
      }),
    ),
  );
  surface.renderOrder = 2;
  plane.add(surface);
  const border = new T.LineSegments(
    keep(new T.EdgesGeometry(surface.geometry)),
    keep(
      new T.LineBasicMaterial({
        color: 0x657f99,
        transparent: true,
        opacity: 0.26,
        depthWrite: false,
      }),
    ),
  );
  plane.add(border);
  plane.add(
    label("Picture plane", new T.Vector3(0.28, 0, -0.48), 0.027).sprite,
  );

  const axes = new T.Group();
  axes.name = "reference-frame-axes";
  axes.matrixAutoUpdate = false;
  axes.visible = false;
  root.add(axes);
  const axisLabels = ["X", "Y", "Z"].map((name, index) => {
    const direction = new T.Vector3().setComponent(index, 1);
    axes.add(
      arrow(
        direction,
        new T.Vector3(),
        0.19,
        [0xc04444, 0x37804e, 0x3b68ad][index]!,
      ),
    );
    const caption = label(name, direction.multiplyScalar(0.225));
    axes.add(caption.sprite);
    return caption;
  });
  const marker = new T.Mesh(
    keep(new T.SphereGeometry(0.008, 12, 8)),
    keep(
      new T.MeshBasicMaterial({
        color: 0x344a53,
        depthTest: false,
        depthWrite: false,
      }),
    ),
  );
  marker.renderOrder = 13;
  axes.add(marker);
  const axisOriginLabel = label(
    "Origin",
    new T.Vector3(-0.025, -0.026, 0),
    0.027,
  );
  axes.add(axisOriginLabel.sprite);

  const movements = new Map<keyof JointSettings, T.Group>();
  const directionLabels: {
    key: keyof JointSettings;
    sign: 1 | -1;
    label: ReturnType<typeof label>;
  }[] = [];
  for (const key of Object.keys(
    labelsByMode.common,
  ) as (keyof JointSettings)[]) {
    const group = new T.Group();
    group.name = `movement-${key}`;
    group.matrixAutoUpdate = false;
    group.visible = false;
    root.add(group);
    movements.set(key, group);
    for (const sign of [-1, 1] as const) {
      const colour = sign > 0 ? 0xa4442d : 0x476c85;
      let labelPosition: T.Vector3;
      if (
        key === "insertionMm" ||
        key === "liftMm" ||
        key === "longitudinalMm"
      ) {
        const direction = translationDirection(key, sign);
        group.add(
          arrow(
            direction,
            direction.clone().multiplyScalar(0.035),
            0.25,
            colour,
          ),
        );
        labelPosition = direction.clone().multiplyScalar(0.33);
      } else {
        const axis =
          key === "angulationDeg"
            ? new T.Vector3(1, 0, 0)
            : new T.Vector3(0, 0, 1);
        const arc = rotationArc(
          axis,
          new T.Vector3(0, 1, 0),
          sign,
          key === "orbitalDeg" ? 0.44 : 0.27,
        );
        const line = new T.Line(
          keep(new T.BufferGeometry().setFromPoints(arc.points)),
          keep(
            new T.LineBasicMaterial({
              color: colour,
              depthTest: false,
              depthWrite: false,
              transparent: true,
              opacity: 0.92,
            }),
          ),
        );
        line.renderOrder = 12;
        group.add(line);
        // The straight helper is only the terminal tangent, not the arc itself.
        group.add(
          arrow(
            arc.tangent,
            arc.end.clone().addScaledVector(arc.tangent, -0.035),
            0.04,
            colour,
          ),
        );
        labelPosition = arc.end.clone().multiplyScalar(1.16);
      }
      const caption = label(
        labelsByMode.common[key][sign > 0 ? "positive" : "negative"],
        labelPosition,
      );
      group.add(caption.sprite);
      directionLabels.push({ key, sign, label: caption });
    }
  }
  let lastMode: TerminologyMode | null = null;
  let lastReference: ReferenceFrame | null = null;
  const detectorBasis = new T.Matrix4().makeBasis(
    new T.Vector3(1, 0, 0),
    new T.Vector3(0, 0, -1),
    new T.Vector3(0, 1, 0),
  );
  return {
    update(state: GuideState) {
      plane.matrix.copy(state.isoMatrix);
      axes.matrix.copy(state.referenceMatrix);
      if (state.reference === "detector") axes.matrix.multiply(detectorBasis);
      for (const [key, group] of movements) {
        group.visible = state.hover === key;
        if (key === "angulationDeg") group.matrix.copy(state.angulationMatrix);
        else if (key === "orbitalDeg") group.matrix.copy(state.orbitalMatrix);
        else {
          group.matrix.identity();
          group.matrix.setPosition(
            state.translationAnchor ??
              new T.Vector3().setFromMatrixPosition(state.isoMatrix),
          );
        }
      }
      if (state.mode !== lastMode) {
        for (const item of directionLabels)
          item.label.setText(
            labelsByMode[state.mode][item.key][
              item.sign > 0 ? "positive" : "negative"
            ],
          );
        lastMode = state.mode;
      }
      if (state.reference !== lastReference) {
        axisLabels.forEach((caption, index) =>
          caption.setText(
            (state.reference === "detector"
              ? ["U", "V", "N"]
              : ["X", "Y", "Z"])[index]!,
          ),
        );
        axisOriginLabel.setText(
          state.reference === "world"
            ? "World origin"
            : state.reference === "patient"
              ? "Body target"
              : "Detector origin",
        );
        lastReference = state.reference;
      }
      root.updateMatrixWorld(true);
    },
    resizeLabels(camera: T.PerspectiveCamera, viewportHeight: number) {
      // Keep directions readable while zooming; the label anchors stay in 3D.
      camera.updateMatrixWorld();
      const position = new T.Vector3();
      for (const { sprite } of textSprites) {
        sprite
          .getWorldPosition(position)
          .applyMatrix4(camera.matrixWorldInverse);
        const height =
          (2 *
            Math.tan(T.MathUtils.degToRad(camera.fov / 2)) *
            Math.max(camera.near, -position.z) *
            28) /
          Math.max(1, viewportHeight);
        sprite.scale.set((height * 768) / 80, height, 1);
      }
    },
    setPlaneVisible(visible: boolean) {
      plane.visible = visible;
    },
    setAxesVisible(visible: boolean) {
      axes.visible = visible;
    },
    dispose() {
      root.removeFromParent();
      for (const resource of resources) resource.dispose();
      resources.clear();
      textSprites.length = 0;
    },
  };
}
