// ===== ANIMATIONS ENGINE =====
const Anim = {
  // Counter animation
  countUp(el, target, duration = 1200) {
    if (!el) return;
    const start = 0;
    const startTime = performance.now();
    const step = (now) => {
      const elapsed = now - startTime;
      const progress = Math.min(elapsed / duration, 1);
      const ease = 1 - Math.pow(1 - progress, 3);
      el.textContent = Math.round(start + (target - start) * ease).toLocaleString();
      if (progress < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  },

  // Stagger entrance
  staggerIn(selector, delay = 80) {
    document.querySelectorAll(selector).forEach((el, i) => {
      el.style.opacity = '0';
      el.style.transform = 'translateY(16px)';
      el.style.transition = 'all 0.5s cubic-bezier(0.4,0,0.2,1)';
      setTimeout(() => {
        el.style.opacity = '1';
        el.style.transform = 'translateY(0)';
      }, i * delay + 50);
    });
  },

  // Fade in element
  fadeIn(el, delay = 0) {
    if (!el) return;
    el.style.opacity = '0';
    el.style.transform = 'translateY(12px)';
    el.style.transition = 'all 0.5s cubic-bezier(0.4,0,0.2,1)';
    setTimeout(() => {
      el.style.opacity = '1';
      el.style.transform = 'translateY(0)';
    }, delay);
  },

  // Progress bar animation
  animateProgress(el, pct, delay = 200) {
    if (!el) return;
    el.style.width = '0%';
    setTimeout(() => { el.style.width = pct + '%'; }, delay);
  }
};
