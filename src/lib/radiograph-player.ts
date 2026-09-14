interface Frame {
  id: string;
  value: number;
  image: string;
  difference?: string;
}
interface Group {
  id: string;
  label: string;
  unit: string;
  initial: number;
  frames: Frame[];
}

/** Record playback only. No image interpolation or projection calculation. */
class RecordedRadiographPlayer extends HTMLElement {
  private cleanup: (() => void) | undefined;
  connectedCallback() {
    if (this.cleanup) return;
    const groups = JSON.parse(this.dataset.groups ?? "[]") as Group[];
    const axis = this.querySelector<HTMLSelectElement>("[data-axis]");
    const slider = this.querySelector<HTMLInputElement>("[data-frame]");
    const play = this.querySelector<HTMLButtonElement>("[data-play]");
    const reset = this.querySelector<HTMLButtonElement>("[data-reset]");
    const output = this.querySelector<HTMLOutputElement>("[data-value]");
    const status = this.querySelector<HTMLElement>("[data-status]");
    const picture = this.querySelector<HTMLImageElement>("[data-frame-image]");
    const difference = this.querySelector<HTMLImageElement>(
      "[data-frame-difference]",
    );
    if (
      !axis ||
      !slider ||
      !play ||
      !reset ||
      !output ||
      !status ||
      !picture ||
      !groups.length
    )
      return;
    let group = groups.find((g) => g.id === axis.value) ?? groups[0]!;
    const retainedIndex = Number(slider.value);
    let index =
        Number.isInteger(retainedIndex) && group.frames[retainedIndex]
          ? retainedIndex
          : group.initial,
      direction = 1,
      active = false,
      visible = true,
      generation = 0;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const reduced = matchMedia("(prefers-reduced-motion: reduce)");
    const controller = new AbortController();
    const { signal } = controller;
    const base = this.dataset.base ?? "";
    const stop = () => {
      active = false;
      ++generation;
      clearTimeout(timer);
      axis.value = group.id;
      slider.max = String(group.frames.length - 1);
      slider.value = String(index);
      play.textContent = "Play poses";
      play.setAttribute("aria-pressed", "false");
    };
    const format = (f: Frame) =>
      `${f.value > 0 ? "+" : ""}${Number(f.value.toPrecision(5))} ${group.unit}`;
    const load = async (file: string) => {
      if (
        !/^[a-zA-Z0-9][a-zA-Z0-9._/-]*$/.test(file) ||
        file.split("/").includes("..")
      )
        throw new Error("Invalid image path");
      const image = new Image();
      image.src = base + file;
      await image.decode();
      return image.src;
    };
    const show = async (next: number, targetGroup: Group = group) => {
      const ticket = ++generation;
      const frame = targetGroup.frames[next];
      if (!frame) return false;
      try {
        const [src, delta] = await Promise.all([
          load(frame.image),
          difference && frame.difference
            ? load(frame.difference)
            : Promise.resolve(undefined),
        ]);
        if (ticket !== generation || !this.isConnected) return false;
        group = targetGroup;
        index = next;
        axis.value = group.id;
        slider.max = String(group.frames.length - 1);
        picture.src = src;
        picture.alt = `Recorded synthetic CT projection: ${group.label}, ${format(frame)}.`;
        if (difference && delta) {
          difference.src = delta;
          difference.alt = `Absolute expected-count change from the reference at ${format(frame)}. White denotes larger change.`;
        }
        slider.value = String(index);
        slider.setAttribute("aria-valuetext", format(frame));
        output.value = format(frame);
        status.textContent = `${group.label}: ${format(frame)}. Recorded pose ${index + 1} of ${group.frames.length}.`;
        return true;
      } catch {
        if (ticket === generation) {
          stop();
          status.textContent =
            "This recorded image could not load. The previous image is retained. Choose a pose to retry.";
          slider.value = String(index);
        }
        return false;
      }
    };
    const tick = async () => {
      if (!active || !visible || document.hidden) return stop();
      if (index + direction >= group.frames.length || index + direction < 0)
        direction *= -1;
      const pending = show(index + direction);
      const ticket = generation;
      const shown = await pending;
      if (ticket !== generation) return;
      if (shown && active) timer = setTimeout(tick, 180);
      else if (!shown) stop();
    };
    axis.addEventListener(
      "change",
      () => {
        const targetGroup = groups.find((g) => g.id === axis.value)!;
        stop();
        void show(targetGroup.initial, targetGroup);
      },
      { signal },
    );
    slider.addEventListener(
      "input",
      () => {
        const next = Number(slider.value);
        stop();
        void show(next);
      },
      { signal },
    );
    reset.addEventListener(
      "click",
      () => {
        stop();
        void show(group.initial);
      },
      { signal },
    );
    play.addEventListener(
      "click",
      () => {
        if (active) return stop();
        if (reduced.matches) {
          status.textContent =
            "Reduced motion is enabled. Use the recorded-pose slider to step through the images.";
          return;
        }
        active = true;
        play.textContent = "Pause";
        play.setAttribute("aria-pressed", "true");
        void tick();
      },
      { signal },
    );
    document.addEventListener(
      "visibilitychange",
      () => {
        if (document.hidden) stop();
      },
      { signal },
    );
    reduced.addEventListener(
      "change",
      () => {
        if (reduced.matches) stop();
      },
      { signal },
    );
    const observer = new IntersectionObserver((entries) => {
      visible = entries[0]?.isIntersecting ?? false;
      if (!visible) stop();
    });
    observer.observe(this);
    for (const control of [axis, slider, play, reset]) control.disabled = false;
    slider.setAttribute("aria-valuetext", format(group.frames[index]!));
    play.setAttribute("aria-pressed", "false");
    this.cleanup = () => {
      stop();
      ++generation;
      controller.abort();
      observer.disconnect();
      this.cleanup = undefined;
    };
  }
  disconnectedCallback() {
    this.cleanup?.();
  }
}
if (!customElements.get("recorded-radiograph-player"))
  customElements.define("recorded-radiograph-player", RecordedRadiographPlayer);
