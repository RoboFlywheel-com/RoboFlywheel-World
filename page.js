(() => {
  "use strict";
  const data = window.RFW_CATALOG;
  const $ = id => document.getElementById(id);
  const menu = $("navigation"), toggle = $("menu-toggle");
  function menuOpen(open) {
    menu.classList.toggle("open", open);
    toggle.setAttribute("aria-expanded", String(open));
  }
  toggle.onclick = () => menuOpen(!menu.classList.contains("open"));
  menu.onclick = () => menuOpen(false);
  document.addEventListener("click", e => { if (!e.target.closest(".site-header")) menuOpen(false); });
  document.addEventListener("keydown", e => {
    if (e.key === "Escape" && menu.classList.contains("open")) { menuOpen(false); toggle.focus(); }
  });
  const navObserver = new IntersectionObserver(entries => {
    for (const entry of entries) if (entry.isIntersecting) {
      for (const a of menu.querySelectorAll("a")) a.classList.toggle("current", a.hash === "#" + entry.target.id);
    }
  }, {rootMargin: "-10% 0px -60%"});
  for (const a of menu.querySelectorAll("a[href^='#']")) {
    const target = document.querySelector(a.hash);
    if (target) navObserver.observe(target);
  }
  if (!data) return;
  const media = "media/";
  const make = (tag, cls, text) => {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text !== undefined) node.textContent = text;
    return node;
  };
  const picture = (file, alt, cls = "") => {
    const img = make("img", cls);
    img.src = media + file;
    img.alt = alt;
    return img;
  };
  const file = (asset, mode) => asset[mode] + ".webp";
  const pressed = (button, active) => {
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", String(active));
  };
  const dialog = $("viewer");
  const hero = $("hero-video"), reduced = matchMedia("(prefers-reduced-motion: reduce)");
  const demo = media + "film.mp4";
  let playRequested = !reduced.matches && !navigator.connection?.saveData, visible = true;
  function syncHero() {
    if (visible && playRequested && !document.hidden && !dialog.open) {
      if (!hero.hasAttribute("src")) hero.src = demo;
      hero.play().catch(() => {});
    } else hero.pause();
  }
  const syncLabel = () => { $("hero-play").textContent = hero.paused ? "Play demo" : "Pause demo"; };
  hero.onplay = hero.onpause = syncLabel;
  $("hero-play").onclick = () => { playRequested = hero.paused; syncHero(); };
  function watch(on) {
    $("home").classList.toggle("is-watching", on);
    $("hero-paper-info").inert = on;
    $("hero-view").setAttribute("aria-expanded", String(!on));
    $("hero-view").textContent = on ? "Show project info ↙" : "Watch demo ↗";
    if (on) { playRequested = true; syncHero(); }
  }
  $("hero-view").onclick = () => watch(!$("hero-paper-info").inert);
  document.addEventListener("keydown", e => {
    if (e.key === "Escape" && $("hero-paper-info").inert && !dialog.open) { watch(false); $("hero-view").focus(); }
  });
  new IntersectionObserver(([entry]) => { visible = entry.isIntersecting; syncHero(); }).observe(hero);
  document.addEventListener("visibilitychange", syncHero);
  reduced.addEventListener("change", () => { if (reduced.matches) { playRequested = false; syncHero(); } });
  function open(label, title, ...nodes) {
    hero.pause();
    $("scene-video").pause();
    $("dialog-label").textContent = label;
    $("dialog-title").textContent = title;
    $("dialog-body").replaceChildren(...nodes);
    dialog.showModal();
    $("close").focus();
  }
  $("close").onclick = () => dialog.close();
  dialog.onclick = e => {
    const r = dialog.getBoundingClientRect();
    if (e.target === dialog && (e.clientX < r.left || e.clientX > r.right ||
        e.clientY < r.top || e.clientY > r.bottom)) dialog.close();
  };
  dialog.onclose = () => {
    $("dialog-body").replaceChildren();
    dialog.classList.remove("film-dialog");
    syncHero();
  };
  $("film").onclick = () => {
    dialog.classList.add("film-dialog");
    const video = make("video");
    video.controls = video.playsInline = true;
    video.src = demo;
    open("RoboFlyWheel", "RFW-W project demo", video);
    video.play().catch(() => {});
  };
  const stage = $("object-view");
  const hullView = $("hull-view"), viewport = $("mesh-viewport"), split = $("mesh-split");
  let active = data.assets[0], currentMode = "visual", share = 50;
  function setSplit(value) {
    share = Math.max(0, Math.min(100, Math.round(value)));
    viewport.style.setProperty("--split", share + "%");
    split.setAttribute("aria-valuenow", share);
    split.setAttribute("aria-valuetext", share + "% appearance");
  }
  function dragSplit(event) {
    const rect = viewport.getBoundingClientRect();
    setSplit((event.clientX - rect.left) * 100 / rect.width);
  }
  split.onpointerdown = e => { split.setPointerCapture(e.pointerId); dragSplit(e); };
  split.onpointermove = e => { if (split.hasPointerCapture(e.pointerId)) dragSplit(e); };
  split.onkeydown = e => {
    const values = {ArrowLeft: share - 1, ArrowRight: share + 1, Home: 0, End: 100};
    if (Object.hasOwn(values, e.key)) { e.preventDefault(); setSplit(values[e.key]); }
  };
  // Both GLBs share export coordinates. Copy the actual lens and orbit, not
  // relative zoom percentages, which depend on each mesh's bounding box.
  function syncCamera() {
    if (currentMode !== "compare" || !stage.loaded || !hullView.loaded) return;
    const {theta, phi, radius} = stage.getCameraOrbit();
    const {x, y, z} = stage.getCameraTarget();
    hullView.cameraOrbit = theta + "rad " + phi + "rad " + radius + "m";
    hullView.cameraTarget = x + "m " + y + "m " + z + "m";
    // model-viewer expands the requested vertical FOV in a narrow viewport.
    const aspect = hullView.clientWidth / hullView.clientHeight;
    const framing = Math.max(1, hullView.getIdealAspect() / aspect);
    hullView.fieldOfView = (Math.atan(Math.tan(stage.getFieldOfView() * Math.PI / 360) / framing) * 360 / Math.PI) + "deg";
    hullView.updateComplete.then(() => hullView.jumpCameraToGoal());
  }
  stage.addEventListener("camera-change", syncCamera);
  hullView.addEventListener("load", syncCamera);
  const loader = document.createElement("script");
  loader.type = "module";
  loader.src = media + "viewer.js";
  loader.onerror = () => { $("viewer-status").hidden = false; };
  document.head.append(loader);
  stage.addEventListener("error", () => {
    $("viewer-status").hidden = false;
    stage.showPoster?.();
  });
  stage.addEventListener("load", () => { $("viewer-status").hidden = true; syncCamera(); });
  function inspect(asset, mode = currentMode) {
    active = asset;
    currentMode = mode;
    const comparing = mode === "compare", primary = comparing ? "visual" : mode;
    hullView.hidden = $("split-tools").hidden = !comparing;
    if (comparing) {
      hullView.poster = media + asset.collision + ".webp";
      hullView.setAttribute("src", media + asset.collision + ".glb");
    } else hullView.removeAttribute("src");
    const poster = media + asset[primary] + ".webp";
    $("object-name").textContent = asset.name;
    const facts = [["Family", asset.family]];
    if (asset.size_cm) facts.push(["Mesh bounds · X × Y × Z", asset.size_cm.map(n => n.toLocaleString("en-US")).join(" × ") + " cm"]);
    if (comparing) facts.push(["Triangles · appearance / collision",
      [asset.visual_triangles, asset.collision_triangles].map(n => n.toLocaleString("en-US")).join(" / ")]);
    else if (asset[primary + "_triangles"]) facts.push(["Triangles · " + (primary === "visual" ? "appearance" : "collision"),
      asset[primary + "_triangles"].toLocaleString("en-US")]);
    facts.push(["Collision hulls", asset.hulls]);
    $("object-facts").replaceChildren(...facts.map(([label, value]) => {
      const cell = make("div"); cell.append(make("dt", "", label), make("dd", "", value)); return cell;
    }));
    $("object-poster").src = poster;
    stage.setAttribute("alt", asset.name + (comparing ? ", comparing appearance and collision hulls" : primary === "visual" ? ", appearance" : ", collision hulls"));
    stage.setAttribute("poster", poster);
    stage.setAttribute("src", media + asset[primary] + ".glb");
    if (comparing) syncCamera();
    for (const b of $("view-modes").children) pressed(b, b.dataset.mode === mode);
    for (const b of $("asset-grid").children) pressed(b, b.dataset.key === asset.visual);
  }
  for (const b of $("view-modes").children) b.onclick = () => inspect(active, b.dataset.mode);
  let chosenFamily = "All", limit = 18;
  function display(family, more = false) {
    limit = more ? limit + 18 : 18;
    chosenFamily = family;
    const grid = $("asset-grid");
    grid.replaceChildren();
    const query = $("search").value.trim().toLowerCase();
    const selected = data.assets.filter(a => (family === "All" || a.family === family) &&
      [a.name, a.family].some(value => value.toLowerCase().includes(query)));
    $("empty").hidden = selected.length !== 0;
    $("count").textContent = "Showing " + Math.min(limit, selected.length) + " of " + selected.length + " matching assets";
    $("more-assets").hidden = limit >= selected.length;
    for (const btn of $("filters").children) pressed(btn, btn.textContent === family);
    for (const asset of selected.slice(0, limit)) {
      const button = make("button", "asset-card");
      button.setAttribute("aria-label", "Inspect " + asset.name);
      const plate = make("span", "asset-image");
      plate.append(picture(file(asset, "visual"), ""));
      const line = make("span", "asset-info");
      line.append(make("strong", "asset-title", asset.name), make("small", "asset-family", asset.family));
      button.append(plate, line);
      button.dataset.key = asset.visual;
      pressed(button, active === asset);
      button.onclick = () => {
        inspect(asset);
        $("inspection").scrollIntoView({block: "center", behavior: "instant"});
        stage.focus({preventScroll: true});
      };
      grid.append(button);
    }
  }
  $("search").oninput = () => display(chosenFamily);
  $("more-assets").onclick = () => display(chosenFamily, true);
  $("reset").onclick = () => { $("search").value = ""; display("All"); $("search").focus(); };
  const families = new Set(data.assets.map(a => a.family));
  for (const [value, caption] of [[data.assets.length, "Featured assets"], [families.size, "Showcase families"],
    [2, "3D representations per asset"], [data.scenes.length, "Scene recordings"]]) {
    const stat = make("div", "stat");
    stat.append(make("strong", "", value), make("span", "", caption));
    $("stats").append(stat);
  }
  for (const name of ["All", ...families]) {
    const button = make("button", "chip", name);
    button.onclick = () => display(name);
    $("filters").append(button);
  }
  function scene(s) {
    const video = $("scene-video");
    video.pause();
    video.poster = media + s.id + ".webp";
    video.src = media + s.id + ".mp4";
    video.setAttribute("aria-label", s.task + " in " + s.name);
    $("scene-title").textContent = s.task;
    $("scene-scope").textContent = s.scope;
    for (const b of $("scene-list").children) pressed(b, b.dataset.id === s.id);
  }
  for (const s of data.scenes) {
    const b = make("button", "action-option"), title = make("span", "", s.task + " · " + s.name);
    b.dataset.id = s.id;
    b.append(picture(s.id + ".webp", ""), title);
    b.onclick = () => scene(s);
    $("scene-list").append(b);
  }
  const first = data.scenes[0];
  inspect(data.assets[0]);
  display("All");
  scene(first);
  for (const el of document.querySelectorAll(".offline")) el.hidden = true;
  for (const id of ["film", "hero-playback", "asset-content", "scene-content"]) $(id).hidden = false;
})();
