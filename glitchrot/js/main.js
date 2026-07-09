const canvas = document.getElementById("particleCanvas");
const ctx = canvas.getContext("2d", { alpha: true });
const particles = [];
const glyphs = "01027";

function resizeCanvas() {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.floor(window.innerWidth * dpr);
  canvas.height = Math.floor(window.innerHeight * dpr);
  canvas.style.width = `${window.innerWidth}px`;
  canvas.style.height = `${window.innerHeight}px`;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

function seedParticles() {
  particles.length = 0;
  const count = Math.round(window.innerWidth / 28);
  for (let i = 0; i < count; i += 1) {
    particles.push({
      x: Math.random() * window.innerWidth,
      y: Math.random() * window.innerHeight,
      speed: 0.16 + Math.random() * 0.42,
      alpha: 0.03 + Math.random() * 0.13,
      size: 9 + Math.random() * 12,
      pulse: Math.random() * Math.PI * 2,
      glyph: glyphs[Math.floor(Math.random() * glyphs.length)]
    });
  }
}

function drawParticles(time = 0) {
  ctx.clearRect(0, 0, window.innerWidth, window.innerHeight);
  ctx.font = "15px Consolas, monospace";
  ctx.textAlign = "center";

  particles.forEach((particle) => {
    const flicker = Math.sin(time * 0.0018 + particle.pulse) * 0.17;
    ctx.fillStyle = `rgba(184,255,0,${Math.max(0.04, particle.alpha + flicker)})`;
    ctx.shadowColor = "rgba(184,255,0,0.8)";
    ctx.shadowBlur = 2;
    ctx.fillText(particle.glyph, particle.x, particle.y);

    particle.y += particle.speed;
    if (particle.y > window.innerHeight + particle.size) {
      particle.y = -20;
      particle.x = Math.random() * window.innerWidth;
      particle.glyph = glyphs[Math.floor(Math.random() * glyphs.length)];
    }
  });

  requestAnimationFrame(drawParticles);
}

function bindUploadTerminal() {
  const dropZone = document.querySelector(".drop-zone");
  const analyze = document.querySelector(".analyze-btn");
  const status = document.querySelector(".status-block strong");
  const statusNote = document.querySelector(".status-block span");

  const armTerminal = () => {
    dropZone.classList.add("is-armed");
    status.textContent = "IMAGE QUEUED";
    statusNote.textContent = "Synthetic trace scan ready";
  };

  dropZone.addEventListener("click", armTerminal);
  dropZone.addEventListener("dragover", (event) => {
    event.preventDefault();
    dropZone.classList.add("is-armed");
  });
  dropZone.addEventListener("dragleave", () => dropZone.classList.remove("is-armed"));
  dropZone.addEventListener("drop", (event) => {
    event.preventDefault();
    armTerminal();
  });

  analyze.addEventListener("click", () => {
    status.textContent = "SCANNING";
    statusNote.textContent = "Bio-forensic engine active";
    document.body.classList.add("analysis-pulse");
    window.setTimeout(() => document.body.classList.remove("analysis-pulse"), 1250);
  });
}

window.addEventListener("resize", () => {
  resizeCanvas();
  seedParticles();
});

resizeCanvas();
seedParticles();
drawParticles();
bindUploadTerminal();
