// A slowly morphing, polished 3D stone rendered with a single WebGL fragment
// shader (signed-distance ray marching). No dependencies, no network.
// Drag to turn it; it keeps drifting on its own.

const VERTEX_SHADER = `
attribute vec2 aPosition;
void main() { gl_Position = vec4(aPosition, 0.0, 1.0); }
`;

const FRAGMENT_SHADER = `
precision highp float;
uniform vec2 uResolution;
uniform float uTime;
uniform float uEnergy;
uniform mat3 uRotation;
uniform vec3 uAccent;
uniform vec3 uStone;
uniform vec2 uPointer;

float hash(vec3 p) {
  p = fract(p * 0.3183099 + vec3(0.71, 0.113, 0.419));
  p *= 17.0;
  return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
}

float noise(vec3 x) {
  vec3 i = floor(x);
  vec3 f = fract(x);
  f = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
  return mix(
    mix(mix(hash(i), hash(i + vec3(1, 0, 0)), f.x),
        mix(hash(i + vec3(0, 1, 0)), hash(i + vec3(1, 1, 0)), f.x), f.y),
    mix(mix(hash(i + vec3(0, 0, 1)), hash(i + vec3(1, 0, 1)), f.x),
        mix(hash(i + vec3(0, 1, 1)), hash(i + vec3(1, 1, 1)), f.x), f.y),
    f.z);
}

// Distance to the stone: a pebble whose surface flows like slow liquid.
float stone(vec3 p) {
  vec3 q = uRotation * p;
  float t = uTime;
  float flow = 0.34 + 0.16 * uEnergy;
  float swell = noise(q * 1.15 + vec3(0.0, t * 0.21, t * 0.13)) - 0.5;
  float ripple = noise(q * 2.6 - vec3(t * 0.27, 0.0, t * 0.19)) - 0.5;
  vec3 squash = q * vec3(1.0, 1.16, 1.07);
  return (length(squash) - 1.0 - swell * flow * 1.25 - ripple * flow * 0.34) * 0.62;
}

vec3 normalAt(vec3 p) {
  vec2 e = vec2(0.0025, -0.0025);
  return normalize(
    e.xyy * stone(p + e.xyy) + e.yyx * stone(p + e.yyx) +
    e.yxy * stone(p + e.yxy) + e.xxx * stone(p + e.xxx));
}

// A soft studio: a large overhead softbox, a side strip and a dim floor.
vec3 studio(vec3 d) {
  float top = smoothstep(0.35, 0.95, d.y) * smoothstep(-0.9, 0.2, d.z + d.x * 0.3);
  float strip = smoothstep(0.86, 0.985, dot(d, normalize(vec3(-0.8, 0.25, 0.45))));
  float floorGlow = smoothstep(-0.2, -1.0, d.y) * 0.10;
  vec3 color = vec3(0.045, 0.05, 0.055);
  color += vec3(1.0, 0.97, 0.92) * top * 0.75;
  color += mix(vec3(1.0), uAccent, 0.75) * strip * 1.6;
  color += uAccent * floorGlow;
  return color;
}

void main() {
  vec2 uv = (gl_FragCoord.xy * 2.0 - uResolution) / min(uResolution.x, uResolution.y);
  vec3 origin = vec3(0.0, 0.0, 3.6);
  vec3 dir = normalize(vec3(uv * 0.86, -2.0));

  float travelled = 1.6;
  float nearest = 10.0;
  bool hit = false;
  for (int i = 0; i < 72; i++) {
    float d = stone(origin + dir * travelled);
    nearest = min(nearest, d);
    if (d < 0.0015) { hit = true; break; }
    travelled += d;
    if (travelled > 6.0) break;
  }

  // Soft shadow pooled under the stone.
  vec2 ground = uv - vec2(0.0, -1.02);
  float shadow = exp(-dot(ground * vec2(1.0, 4.6), ground * vec2(1.0, 4.6)) * 1.4) * 0.34;
  vec4 color = vec4(0.0, 0.0, 0.0, shadow);

  if (hit) {
    vec3 p = origin + dir * travelled;
    vec3 n = normalAt(p);
    vec3 q = uRotation * p;

    // Mineral veining that travels with the stone as it turns.
    float bands = noise(q * 3.1 + noise(q * 1.7) * 2.4);
    float vein = smoothstep(0.035, 0.0, abs(bands - 0.5)) * 0.85;
    float speckle = noise(q * 26.0);
    vec3 albedo = pow(uStone, vec3(2.2)) * (0.6 + 0.9 * noise(q * 2.2));
    albedo = mix(albedo, mix(vec3(0.75, 0.73, 0.68), uAccent, 0.3), vein * 0.5);
    albedo *= 0.9 + 0.2 * speckle;

    vec3 keyDir = normalize(vec3(0.55 + uPointer.x * 0.5, 0.8 + uPointer.y * 0.3, 0.65));
    float key = max(dot(n, keyDir), 0.0);
    float fill = max(dot(n, normalize(vec3(-0.7, -0.15, 0.5))), 0.0);
    float occlusion = 0.55 + 0.45 * smoothstep(-0.9, 0.6, n.y);

    vec3 view = -dir;
    float fresnel = pow(1.0 - max(dot(n, view), 0.0), 4.0);
    vec3 reflection = studio(reflect(dir, n));
    float gloss = 0.16 + 0.22 * (1.0 - speckle * 0.5) + vein * 0.2;
    vec3 halfway = normalize(keyDir + view);
    float sparkle = pow(max(dot(n, halfway), 0.0), 90.0);

    vec3 lit = albedo * (key * vec3(1.0, 0.96, 0.9) * 2.6 + fill * uAccent * 0.5 + 0.22) * occlusion;
    lit += reflection * (gloss * 0.12 + fresnel * 0.55);
    lit += vec3(1.0) * sparkle * 0.9;
    lit += uAccent * pow(1.0 - max(dot(n, view), 0.0), 3.0) * 0.12;

    lit = lit / (lit + vec3(0.6));       // filmic roll-off
    lit = pow(lit, vec3(0.4545));         // to display gamma
    color = vec4(lit, 1.0);
  } else {
    // A faint halo in the accent colour around the silhouette.
    float halo = exp(-nearest * 6.5) * 0.11;
    color.rgb = uAccent * halo;
    color.a = max(color.a, halo);
    // Soften the silhouette: rays that graze the stone get partial coverage.
    float pixel = 3.1 / min(uResolution.x, uResolution.y);
    float graze = 1.0 - smoothstep(0.0, pixel * 1.4, nearest / 0.62);
    color = mix(color, vec4(uStone * 0.6, 1.0), graze);
  }
  gl_FragColor = color;   // premultiplied alpha
}
`;

function compile(gl, type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(shader);
    gl.deleteShader(shader);
    throw new Error("Stone shader failed to compile: " + log);
  }
  return shader;
}

function rotationMatrix(yaw, pitch) {
  const cy = Math.cos(yaw), sy = Math.sin(yaw);
  const cp = Math.cos(pitch), sp = Math.sin(pitch);
  // Column-major mat3 = Rx(pitch) * Ry(yaw).
  return new Float32Array([
    cy, sp * sy, -cp * sy,
    0, cp, sp,
    sy, -sp * cy, cp * cy
  ]);
}

function parseColor(value, fallback) {
  const match = /^#([0-9a-f]{6})$/i.exec((value || "").trim());
  if (!match) return fallback;
  const number = parseInt(match[1], 16);
  return [(number >> 16 & 255) / 255, (number >> 8 & 255) / 255, (number & 255) / 255];
}

/**
 * Start a stone on `canvas`. Returns a controller, or null when WebGL is not
 * available (callers keep their static fallback).
 */
export function createStone(canvas, { accent = "#b9f26b", stone = "#3b4038", maxPixels = 420_000 } = {}) {
  if (!canvas || typeof canvas.getContext !== "function") return null;
  let gl;
  try {
    gl = canvas.getContext("webgl", { alpha: true, premultipliedAlpha: true, antialias: false, powerPreference: "low-power" });
  } catch {
    gl = null;
  }
  if (!gl) return null;

  let program;
  try {
    program = gl.createProgram();
    gl.attachShader(program, compile(gl, gl.VERTEX_SHADER, VERTEX_SHADER));
    gl.attachShader(program, compile(gl, gl.FRAGMENT_SHADER, FRAGMENT_SHADER));
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program));
  } catch (error) {
    console.info("3B taş animasyonu başlatılamadı; sabit görsel kullanılıyor.", error);
    return null;
  }
  gl.useProgram(program);
  const buffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
  const position = gl.getAttribLocation(program, "aPosition");
  gl.enableVertexAttribArray(position);
  gl.vertexAttribPointer(position, 2, gl.FLOAT, false, 0, 0);
  gl.clearColor(0, 0, 0, 0);

  const uniform = (name) => gl.getUniformLocation(program, name);
  const uniforms = {
    resolution: uniform("uResolution"), time: uniform("uTime"), energy: uniform("uEnergy"),
    rotation: uniform("uRotation"), accent: uniform("uAccent"), stone: uniform("uStone"),
    pointer: uniform("uPointer")
  };

  const reducedMotion = globalThis.matchMedia?.("(prefers-reduced-motion: reduce)");
  const view = {
    yaw: 0.6, pitch: -0.25, spinX: 0, spinY: 0,
    time: Math.random() * 40, energy: 0, energyTarget: 0,
    pointerX: 0, pointerY: 0, dragging: false, lastX: 0, lastY: 0,
    visible: true, running: false, frame: 0, last: 0, destroyed: false,
    accent: parseColor(accent, [0.73, 0.95, 0.42]), stone: parseColor(stone, [0.23, 0.25, 0.22])
  };

  function resize() {
    const rect = canvas.getBoundingClientRect();
    if (rect.width < 2 || rect.height < 2) return false;
    let ratio = Math.min(globalThis.devicePixelRatio || 1, 2);
    const pixels = rect.width * rect.height * ratio * ratio;
    if (pixels > maxPixels) ratio *= Math.sqrt(maxPixels / pixels);
    const width = Math.max(2, Math.round(rect.width * ratio));
    const height = Math.max(2, Math.round(rect.height * ratio));
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    gl.viewport(0, 0, width, height);
    return true;
  }

  function draw(now) {
    view.frame = 0;
    if (view.destroyed) return;
    const delta = Math.min(0.05, view.last ? (now - view.last) / 1000 : 0.016);
    view.last = now;
    const still = reducedMotion?.matches && !view.dragging;

    view.energy += (view.energyTarget - view.energy) * Math.min(1, delta * 2.5);
    if (!still) {
      view.time += delta * (1 + view.energy * 1.6);
      if (!view.dragging) {
        view.yaw += (0.16 + view.energy * 0.3) * delta + view.spinX * delta;
        view.pitch += view.spinY * delta;
        view.spinX *= Math.pow(0.08, delta);
        view.spinY *= Math.pow(0.08, delta);
        view.pitch += (-0.25 - view.pitch) * Math.min(1, delta * 0.6);
      }
    }

    if (resize()) {
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.uniform2f(uniforms.resolution, canvas.width, canvas.height);
      gl.uniform1f(uniforms.time, view.time);
      gl.uniform1f(uniforms.energy, view.energy);
      gl.uniformMatrix3fv(uniforms.rotation, false, rotationMatrix(view.yaw, view.pitch));
      gl.uniform3fv(uniforms.accent, view.accent);
      gl.uniform3fv(uniforms.stone, view.stone);
      gl.uniform2f(uniforms.pointer, view.pointerX, view.pointerY);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    }
    schedule();
  }

  function schedule() {
    if (view.frame || view.destroyed || !view.running || !view.visible || document.hidden) return;
    view.frame = requestAnimationFrame(draw);
  }

  function onPointerDown(event) {
    view.dragging = true;
    view.lastX = event.clientX;
    view.lastY = event.clientY;
    canvas.setPointerCapture?.(event.pointerId);
    canvas.classList.add("is-dragging");
    schedule();
  }
  function onPointerMove(event) {
    const rect = canvas.getBoundingClientRect();
    view.pointerX = ((event.clientX - rect.left) / rect.width - 0.5) * 2;
    view.pointerY = (0.5 - (event.clientY - rect.top) / rect.height) * 2;
    if (!view.dragging) return;
    const dx = (event.clientX - view.lastX) / Math.max(rect.width, 1);
    const dy = (event.clientY - view.lastY) / Math.max(rect.height, 1);
    view.lastX = event.clientX;
    view.lastY = event.clientY;
    view.yaw += dx * 4.2;
    view.pitch = Math.max(-1.3, Math.min(1.3, view.pitch + dy * 3.2));
    view.spinX = dx * 4.2 * 60;
    view.spinY = dy * 3.2 * 60;
  }
  function onPointerUp(event) {
    view.dragging = false;
    canvas.releasePointerCapture?.(event.pointerId);
    canvas.classList.remove("is-dragging");
  }
  function onVisibility() {
    view.last = 0;
    schedule();
  }

  canvas.addEventListener("pointerdown", onPointerDown);
  canvas.addEventListener("pointermove", onPointerMove);
  canvas.addEventListener("pointerup", onPointerUp);
  canvas.addEventListener("pointercancel", onPointerUp);
  document.addEventListener("visibilitychange", onVisibility);

  let observer = null;
  if ("IntersectionObserver" in globalThis) {
    observer = new IntersectionObserver((entries) => {
      view.visible = entries.some((entry) => entry.isIntersecting);
      view.last = 0;
      schedule();
    });
    observer.observe(canvas);
  }

  return {
    start() {
      view.running = true;
      view.last = 0;
      schedule();
    },
    stop() {
      view.running = false;
      if (view.frame) cancelAnimationFrame(view.frame);
      view.frame = 0;
    },
    /** 0 = calm drift, 1 = lively (used while a photo is being processed). */
    setEnergy(value) {
      view.energyTarget = Math.max(0, Math.min(1, Number(value) || 0));
    },
    setColors({ accent: nextAccent, stone: nextStone } = {}) {
      view.accent = parseColor(nextAccent, view.accent);
      view.stone = parseColor(nextStone, view.stone);
    },
    destroy() {
      this.stop();
      view.destroyed = true;
      observer?.disconnect();
      document.removeEventListener("visibilitychange", onVisibility);
      canvas.removeEventListener("pointerdown", onPointerDown);
      canvas.removeEventListener("pointermove", onPointerMove);
      canvas.removeEventListener("pointerup", onPointerUp);
      canvas.removeEventListener("pointercancel", onPointerUp);
    }
  };
}
