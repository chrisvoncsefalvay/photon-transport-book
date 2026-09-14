export interface ChapterEntry {
  number: string;
  title: string;
  shortTitle: string;
  href: string;
  scope: string;
  longScope: string;
  part: "chapter" | "appendix";
}

export const chapters = [
  {
    number: "01",
    title: "What are we differentiating, and why?",
    shortTitle: "Introduction",
    href: "/chapters/introduction/",
    scope:
      "Defines the questions and boundaries of differentiable photon transport.",
    longScope:
      "Defines the questions and boundaries of differentiable photon transport, beginning with a C-arm and an image, then asking which quantities we want to recover and what a useful derivative would tell us about them.",
    part: "chapter",
  },
  {
    number: "02",
    title: "From photon survival to the transmission law",
    shortTitle: "Transmission",
    href: "/chapters/transmission/",
    scope: "Establishes the transmission model used throughout the volume.",
    longScope:
      "Establishes the transmission model used throughout the volume, following attenuation along a single path and keeping the parameters, units and assumptions visible as that physical account becomes an executable model.",
    part: "chapter",
  },
  {
    number: "03",
    title: "Coordinate frames and the geometry of motion",
    shortTitle: "Frames and motion",
    href: "/chapters/frames-geometry-motion/",
    scope:
      "Fixes the coordinate frames and motion conventions for later chapters.",
    longScope:
      "Fixes the coordinate frames and motion conventions for later chapters, giving the source, detector and object a shared account of where everything is so that a change in pose means the same thing in the geometry and the code.",
    part: "chapter",
  },
  {
    number: "04",
    title: "Tracing rays through a sampled volume",
    shortTitle: "Volumes and rays",
    href: "/chapters/volumes-line-integrals/",
    scope: "Connects volumetric representations with line-integral operators.",
    longScope:
      "Connects volumetric representations with line-integral operators, tracing a ray through a sampled attenuation field and examining how interpolation, boundaries and numerical choices shape the projection we ask the GPU to compute.",
    part: "chapter",
  },
  {
    number: "05",
    title: "Derivatives of the projection, from voxels to pose",
    shortTitle: "Projection gradients",
    href: "/chapters/differentiating-projection/",
    scope: "Develops derivatives of the projection operation.",
    longScope:
      "Develops derivatives of the projection operation, asking how an image changes when its inputs move and how independent numerical checks can distinguish a useful gradient from one that merely looks plausible.",
    part: "chapter",
  },
  {
    number: "06",
    title: "Recovering an object’s pose from its projection",
    shortTitle: "Pose recovery",
    href: "/chapters/recovering-pose/",
    scope: "Studies pose recovery with differentiable image formation.",
    longScope:
      "Studies pose recovery with differentiable image formation, working backwards from an image to geometry while considering the choice of image comparison, the initial estimate and the ambiguities that a single view cannot resolve.",
    part: "chapter",
  },
  {
    number: "07",
    title: "When the X-ray changes but the object has not moved",
    shortTitle: "Acquisition mismatch",
    href: "/chapters/same-pose-different-x-ray/",
    scope:
      "Examines acquisition changes that can confound geometric inference.",
    longScope:
      "Examines acquisition changes that can confound geometric inference, looking at why two X-rays may differ even when nothing has moved and what a recovery method needs to account for before it starts adjusting the object's pose.",
    part: "chapter",
  },
  {
    number: "08",
    title: "From an X-ray spectrum to a detector measurement",
    shortTitle: "Spectra and detectors",
    href: "/chapters/spectral-transport-detector/",
    scope: "Extends the model to spectra and detector response.",
    longScope:
      "Extends the model to spectra and detector response, moving beyond a single photon energy to consider how the source spectrum and the detector's sensitivity affect the predicted image and the parameters we hope to infer from it.",
    part: "chapter",
  },
  {
    number: "09",
    title: "Following scattered photons with Monte Carlo transport",
    shortTitle: "Monte Carlo transport",
    href: "/chapters/scattering-stochastic-transport/",
    scope: "Introduces scattering and stochastic transport estimators.",
    longScope:
      "Introduces scattering and stochastic transport estimators, following photons beyond the straight path from source to detector and considering how sampling choices, variance and computational cost enter the task of estimating an image.",
    part: "chapter",
  },
  {
    number: "10",
    title: "Differentiating transport when photon paths are random",
    shortTitle: "Transport gradients",
    href: "/chapters/differentiating-transport/",
    scope: "Treats derivatives through the full transport process.",
    longScope:
      "Treats derivatives through the full transport process, considering what changes when paths are sampled, which dependencies must be included and how to assess correctness without mistaking sampling noise for a modelling or implementation error.",
    part: "chapter",
  },
  {
    number: "11",
    title: "Registering a CT volume to calibrated radiographs",
    shortTitle: "2D–3D registration",
    href: "/chapters/registration/",
    scope: "Aligns a known volume with measured radiographs.",
    longScope:
      "Aligns a known volume with measured radiographs, extending pose recovery to calibrated acquisitions, multiple views and independent evaluation when the anatomy and image formation do not agree perfectly.",
    part: "chapter",
  },
  {
    number: "12",
    title: "Reconstructing a volume from its X-ray projections",
    shortTitle: "Volume reconstruction",
    href: "/chapters/reconstruction/",
    scope: "Recovers a three-dimensional attenuation field from projections.",
    longScope:
      "Recovers a three-dimensional attenuation field from projections, developing volume optimisation and regularisation before examining sparse views, limited angles and material decomposition with spectral measurements.",
    part: "chapter",
  },
  {
    number: "13",
    title: "Choosing the next view and where to spend the photons",
    shortTitle: "Acquisition design",
    href: "/chapters/acquisition-design/",
    scope: "Chooses measurements that improve registration or reconstruction.",
    longScope:
      "Chooses measurements that improve registration or reconstruction, using task-specific uncertainty and error objectives to select viewpoints and allocate exposure within geometric and acquisition constraints.",
    part: "chapter",
  },
  {
    number: "A",
    title: "X-ray theory for non-radiologists",
    shortTitle: "X-ray physics",
    href: "/appendix/x-ray-theory/",
    scope: "Collects the X-ray concepts required by the main text.",
    longScope:
      "Collects the X-ray concepts required by the main text, keeping physical quantities, imaging terminology and useful conventions within reach for readers who know their way around code but have no reason to know their way around a radiology department.",
    part: "appendix",
  },
] as const satisfies readonly ChapterEntry[];

/** A preview extends the existing sentence instead of replacing its opening. */
export function chapterContinuation(
  scope: string,
  longScope: string,
): {
  prefix: string;
  suffix: string;
} {
  const prefix = scope.replace(/\.$/, "");
  if (!longScope.startsWith(`${prefix}, `)) {
    throw new Error(
      "A chapter's long description must continue its short sentence.",
    );
  }
  return { prefix, suffix: longScope.slice(prefix.length) };
}

export function chapterNeighbours(href: string): {
  previous?: ChapterEntry;
  next?: ChapterEntry;
} {
  const index = chapters.findIndex((chapter) => chapter.href === href);

  if (index < 0) return {};

  return {
    ...(index > 0 ? { previous: chapters[index - 1] } : {}),
    ...(index < chapters.length - 1 ? { next: chapters[index + 1] } : {}),
  };
}
