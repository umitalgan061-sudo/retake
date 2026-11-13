// Liquid particle sculpture — the interactive hero of "Forma — Liquid
// Sculpture" (https://forma-liquid-sculpture-ua.web.app), a 21st.dev community
// component by Guglielmo Giannattasio (https://guglielmogiannattasio.it).
// The WebGL2 engine below (shaders and SculptureEngine) is taken from that
// build, only reformatted and with its top-level names spelled out; the
// mountSculpture() wrapper at the end replaces the original React shell.
//
// A sculpture photo is turned into ~10^5 particles. The pointer blows them
// outward and swirls them around, a liquid "wake" carries them on, curl noise
// adds turbulence and a spring brings every particle home.

function clamp(z, o, h) {
  return Math.min(h, Math.max(o, z));
}
const NOISE_GLSL = /* glsl */ `
float hash(vec2 p){ p = fract(p*vec2(123.34,456.21)); p += dot(p,p+45.32); return fract(p.x*p.y); }
float vnoise(vec2 p){ vec2 i=floor(p); vec2 f=fract(p); vec2 u=f*f*f*(f*(f*6.0-15.0)+10.0);
  return mix(mix(hash(i),hash(i+vec2(1.0,0.0)),u.x), mix(hash(i+vec2(0.0,1.0)),hash(i+vec2(1.0,1.0)),u.x), u.y); }
vec2 ambientField(vec2 uv, float t){
  vec2 q = uv*vec2(3.2,1.7);
  vec2 a = vec2(vnoise(q+vec2(t*0.11,0.0))-0.5, vnoise(q+vec2(5.2,t*0.08)+3.1)-0.5);
  vec2 q2 = uv*vec2(7.5,4.0) + vec2(-t*0.19, t*0.13);
  vec2 b = vec2(vnoise(q2+1.7)-0.5, vnoise(q2+9.3)-0.5);
  return a + b*0.45;
}
`,
  ENERGY_GLSL = `
float energyAt(vec2 rectPx, vec2 mouse, float radius, float presence, float wakeEnergy){
  vec2 d = rectPx - mouse;
  float f = exp(-dot(d,d)/(radius*radius))*presence;
  return smoothstep(0.06, 0.5, f*1.15 + wakeEnergy*1.7);
}
`,
  QUAD_VS = `#version 300 es
layout(location=0) in vec2 aPos;
out vec2 vUv;
void main(){ vUv = aPos*0.5+0.5; gl_Position = vec4(aPos,0.0,1.0); }
`,
  FLOW_FS = `#version 300 es
precision highp float;
in vec2 vUv;
out vec4 o;
uniform sampler2D uPrev;
uniform vec2 uTexel;
uniform vec2 uMouse;
uniform vec2 uVel;
uniform vec2 uSize;
uniform float uRadius;
uniform float uPresence;
uniform float uDecay;
vec2 decV(vec4 c){ return (c.rg*255.0-128.0)/127.0; }
void main(){
  vec2 v0 = decV(texture(uPrev, vUv));
  vec2 uv = vUv - v0*uTexel*2.5;
  vec4 c = texture(uPrev, uv)*0.52
    + 0.12*(texture(uPrev, uv+vec2(uTexel.x,0.0)) + texture(uPrev, uv-vec2(uTexel.x,0.0))
          + texture(uPrev, uv+vec2(0.0,uTexel.y)) + texture(uPrev, uv-vec2(0.0,uTexel.y)));
  vec2 v = decV(c);
  float p = c.b;
  vec2 d = (vUv-uMouse)*uSize;
  float f = exp(-dot(d,d)/(uRadius*uRadius))*uPresence;
  float sp = clamp(length(uVel)*2.5, 0.0, 1.0);
  v += uVel*f*0.22;
  p = max(p, f*sp);
  v *= uDecay;
  p *= uDecay;
  v = sign(v)*max(abs(v)-1.2/127.0, 0.0);
  p = max(p-1.2/255.0, 0.0);
  v = clamp(v, -1.0, 1.0);
  o = vec4((v*127.0+128.0)/255.0, p, 1.0);
}
`,
  SIM_FS = `#version 300 es
precision highp float;
out vec4 o;
uniform sampler2D uState;
uniform sampler2D uHome;
uniform sampler2D uFlow;
uniform vec2 uSize;
uniform vec2 uMouse;
uniform float uRadius;
uniform float uPresence;
uniform float uStrength;
uniform float uSpread;
uniform float uTurb;
uniform float uReturn;
uniform float uTime;
uniform float uDt;
uniform float uIntro;
${NOISE_GLSL}
vec2 curl(vec2 p){
  const float e = 0.35;
  float a = vnoise(p+vec2(0.0,e)) - vnoise(p-vec2(0.0,e));
  float b = vnoise(p+vec2(e,0.0)) - vnoise(p-vec2(e,0.0));
  return vec2(a, -b)/(2.0*e);
}
void main(){
  ivec2 ij = ivec2(gl_FragCoord.xy);
  vec4 hm = texelFetch(uHome, ij, 0);
  if(hm.w <= 0.0){ o = vec4(0.0); return; }
  vec4 st = texelFetch(uState, ij, 0);
  vec2 off = st.xy;
  vec2 vel = st.zw;
  float sd = hm.z;
  float light = 0.45 + sd*1.1;

  vec2 p = hm.xy*uSize + off;
  vec4 fl = texture(uFlow, clamp(p/uSize, 0.0, 1.0));
  vec2 wake = (fl.rg*255.0-128.0)/127.0;
  float pr = fl.b;

  vec2 toM = p - uMouse;
  float r2 = dot(toM,toM);
  float f = exp(-r2/(uRadius*uRadius))*uPresence;
  vec2 n = toM*inversesqrt(r2+1.0);
  float e = clamp(pr*2.0 + f, 0.0, 1.0);

  vec2 acc = vec2(0.0);
  // blown outward and swirled around the cursor
  acc += n*f*1.25*uStrength*uSpread*light;
  acc += vec2(-n.y, n.x)*f*1.45*uStrength*uSpread*(0.4+sd);
  // carried by the liquid wake
  acc += (wake*42.0*uSpread*light - vel)*e*0.08;
  // curl-noise turbulence where the material is disturbed (and during the intro)
  vec2 q = p*0.0095 + vec2(uTime*0.35, -uTime*0.2);
  vec2 c = curl(q) + 0.5*curl(q*2.3 + 7.1);
  acc += c*(e*1.2*uSpread + uIntro*1.5)*uTurb*light;
  // elastic return home
  float k = (0.006 + 0.02*uReturn)*(0.65+sd*0.7)*(1.0 - uIntro*0.75);
  acc -= off*k;

  vel += acc*uDt;
  vel *= pow(mix(0.93, 0.955, e), uDt);
  off += vel*uDt;
  if(dot(off,off) < 0.0004 && dot(vel,vel) < 0.0004){ off = vec2(0.0); vel = vec2(0.0); }
  o = vec4(off, vel);
}
`,
  PARTICLE_VS = `#version 300 es
precision highp float;
layout(location=0) in vec2 aCorner;
uniform sampler2D uState;
uniform sampler2D uHome;
uniform sampler2D uCol;
uniform sampler2D uFlow;
uniform int uTexW;
uniform vec4 uRect;
uniform vec2 uRes;
uniform vec2 uMouse;
uniform float uRadius;
uniform float uPresence;
uniform float uDot;
uniform float uTime;
uniform float uAmbient;
uniform float uStreak;
uniform float uIntroVis;
uniform float uRest;
uniform vec3 uTint;
out vec2 vLocal;
out float vHalfLen;
out float vR;
out vec3 vCol;
out float vA;
${NOISE_GLSL}
${ENERGY_GLSL}
void main(){
  int id = gl_InstanceID;
  ivec2 ij = ivec2(id % uTexW, id / uTexW);
  vec4 hm = texelFetch(uHome, ij, 0);
  vec4 st = texelFetch(uState, ij, 0);
  vec4 cl = texelFetch(uCol, ij, 0);

  float wakeE = texture(uFlow, hm.xy).b;
  float E = energyAt(hm.xy*uRect.zw, uMouse, uRadius, uPresence, wakeE);
  float lift = smoothstep(0.35, 2.5, length(st.xy));
  float a = max(max(E, lift), max(uRest, 1.0-uIntroVis));
  if(a < 0.01 || hm.w <= 0.0){ gl_Position = vec4(2.0,2.0,2.0,1.0); return; }

  vec2 base = uRect.xy + hm.xy*uRect.zw + st.xy;
  base -= ambientField(hm.xy, uTime)*10.0*uAmbient;

  vec2 v = st.zw;
  float sp = length(v);
  vec2 dir = sp > 0.001 ? v/sp : vec2(1.0,0.0);
  float r = uDot*hm.w*0.5;
  float len = min(sp*1.25, 18.0)*uStreak;
  vec2 lc = aCorner*vec2(r + len*0.5 + 1.0, r + 1.0);
  vec2 center = base - dir*len*0.5;
  vec2 world = center + dir*lc.x + vec2(-dir.y, dir.x)*lc.y;

  vec2 clip = world/uRes*2.0-1.0;
  gl_Position = vec4(clip.x, -clip.y, 0.0, 1.0);
  vLocal = lc;
  vHalfLen = len*0.5;
  vR = r;
  vec3 col = cl.rgb*uTint;
  // fast material catches the light, like liquid metal
  float glint = smoothstep(2.0, 16.0, sp);
  col = mix(col, min(col*1.18 + vec3(0.07,0.075,0.085), vec3(1.0)), glint*0.75);
  vCol = col;
  vA = a;
}
`,
  PARTICLE_FS = `#version 300 es
precision highp float;
in vec2 vLocal;
in float vHalfLen;
in float vR;
in vec3 vCol;
in float vA;
uniform float uDpr;
out vec4 o;
void main(){
  vec2 q = vec2(max(abs(vLocal.x)-vHalfLen, 0.0), vLocal.y);
  float d = length(q) - vR;
  float a = clamp(0.5 - d*uDpr, 0.0, 1.0);
  // comet tail: the trailing end thins out
  float tail = vHalfLen > 0.0 ? clamp(0.5 - vLocal.x/(2.0*(vHalfLen+vR)), 0.0, 1.0) : 0.0;
  a *= 1.0 - tail*0.55;
  a *= vA;
  if(a < 0.004) discard;
  o = vec4(vCol*a, a);
}
`,
  IMAGE_FS = `#version 300 es
precision highp float;
out vec4 o;
uniform sampler2D uImg;
uniform sampler2D uFlow;
uniform vec2 uRes;
uniform float uDpr;
uniform vec4 uRect;
uniform vec3 uBg;
uniform vec3 uImgBg;
uniform vec2 uMouse;
uniform float uPresence;
uniform float uRadius;
uniform float uStrength;
uniform float uTime;
uniform float uAmbient;
uniform float uIntroVis;
uniform float uRest;
uniform float uAlphaMode;
${NOISE_GLSL}
${ENERGY_GLSL}
void main(){
  vec2 px = vec2(gl_FragCoord.x, uRes.y*uDpr - gl_FragCoord.y)/uDpr;
  vec2 size = uRect.zw;
  vec2 uv = (px-uRect.xy)/size;
  if(uv.x < -0.05 || uv.x > 1.05 || uv.y < -0.12 || uv.y > 1.12){ o = vec4(0.0); return; }
  vec4 fl = texture(uFlow, clamp(uv,0.0,1.0));
  vec2 vel = (fl.rg*255.0-128.0)/127.0;
  vec2 rp = uv*size;
  vec2 toM = rp - uMouse;
  float fall = exp(-dot(toM,toM)/(uRadius*uRadius))*uPresence;
  vec2 disp = -toM*0.22*uStrength*fall - vel*10.0*uStrength;
  disp += ambientField(uv, uTime)*10.0*uAmbient;
  vec2 suv = uv + disp/size;
  vec4 tx = texture(uImg, suv);
  vec3 col = tx.rgb;
  vec2 e = smoothstep(vec2(0.0), vec2(0.004,0.01), suv) * smoothstep(vec2(1.0), vec2(0.996,0.99), suv);
  // cut-out images carry their own alpha; flat-background images are keyed by colour
  float key = uAlphaMode > 0.5 ? tx.a : smoothstep(0.012, 0.05, length(col-uImgBg));
  float k = key*e.x*e.y;
  float E = energyAt(rp, uMouse, uRadius, uPresence, fl.b);
  float show = (1.0 - max(E, uRest))*uIntroVis;
  vec3 mat = uAlphaMode > 0.5 ? col : col*(uBg/max(uImgBg, vec3(0.001)));
  // premultiplied: empty areas stay transparent so the page (and the backdrop word) shows through
  float A = k*show;
  o = vec4(mat*A, A);
}
`;
function createProgram(z, o, h) {
  const s = (K, D) => {
      const S = z.createShader(K);
      if ((z.shaderSource(S, D), z.compileShader(S), !z.getShaderParameter(S, z.COMPILE_STATUS))) {
        const H = z.getShaderInfoLog(S);
        throw (z.deleteShader(S), new Error("Shader: " + H));
      }
      return S;
    },
    _ = z.createProgram(),
    x = s(z.VERTEX_SHADER, o),
    R = s(z.FRAGMENT_SHADER, h);
  if (
    (z.attachShader(_, x),
    z.attachShader(_, R),
    z.bindAttribLocation(_, 0, "aPos"),
    z.linkProgram(_),
    z.deleteShader(x),
    z.deleteShader(R),
    !z.getProgramParameter(_, z.LINK_STATUS))
  )
    throw new Error("Link: " + z.getProgramInfoLog(_));
  const G = {},
    N = z.getProgramParameter(_, z.ACTIVE_UNIFORMS);
  for (let K = 0; K < N; K++) {
    const D = z.getActiveUniform(_, K);
    D && (G[D.name] = z.getUniformLocation(_, D.name));
  }
  return { p: _, u: G };
}
const STATE_TEX_W = 1024;
const SETTLE_FRAMES = 270;
class SculptureEngine {
  constructor(o, h, s) {
    (this.lastImage = null),
      (this.onFirstFrame = null),
      (this.lowPower = !1),
      (this.stateFormat = { internal: 0, type: 0 }),
      (this.imgTex = null),
      (this.flowTex = []),
      (this.flowFb = []),
      (this.flowW = 256),
      (this.flowH = 104),
      (this.flowIdx = 0),
      (this.flowDirty = !1),
      (this.homeTex = null),
      (this.colTex = null),
      (this.stateTex = []),
      (this.stateFb = []),
      (this.stateIdx = 0),
      (this.texH = 1),
      (this.n = 0),
      (this.gridW = 1),
      (this.stateDirty = !1),
      (this.alphaMode = !1),
      (this.imgBg = [0.918, 0.914, 0.906]),
      (this.flowing = 0),
      (this.aspect = 2.5),
      (this.rect = { x: 0, y: 0, w: 1, h: 1 }),
      (this.cssW = 1),
      (this.cssH = 1),
      (this.target = { x: -9999, y: -9999 }),
      (this.mouse = { x: -9999, y: -9999 }),
      (this.prevMouse = { x: -9999, y: -9999 }),
      (this.mvel = { x: 0, y: 0 }),
      (this.presence = 0),
      (this.presenceTarget = 0),
      (this.burst = 0),
      (this.lastInput = -1e9),
      (this.settle = 1e9),
      (this.touchRelease = 0),
      (this.time = 0),
      (this.introT = 1),
      (this.raf = 0),
      (this.running = !1),
      (this.last = 0),
      (this.skip = 0),
      (this.visible = !0),
      (this.ready = !1),
      (this.destroyed = !1),
      (this.lost = !1),
      (this.onLost = (R) => {
        R.preventDefault(), (this.lost = !0), this.stop(), this.onFail();
      }),
      (this.homeData = null),
      (this.frame = (R) => {
        if (!this.running) return;
        const G = R - this.last,
          N = clamp(G / 16.6667, 0.25, 3);
        this.last = R;
        const K = this.opts,
          D = K.ambient && !K.reducedMotion;
        this.touchRelease && R > this.touchRelease && ((this.presenceTarget = 0), (this.touchRelease = 0));
        const S = 1 - Math.pow(1 - 0.22, N);
        (this.mouse.x += (this.target.x - this.mouse.x) * S),
          (this.mouse.y += (this.target.y - this.mouse.y) * S);
        const H = (this.mouse.x - this.prevMouse.x) / N,
          _t = (this.mouse.y - this.prevMouse.y) / N;
        (this.prevMouse.x = this.mouse.x),
          (this.prevMouse.y = this.mouse.y),
          (this.mvel.x += (clamp(H, -60, 60) - this.mvel.x) * 0.5),
          (this.mvel.y += (clamp(_t, -60, 60) - this.mvel.y) * 0.5);
        const F = this.presenceTarget > this.presence ? 0.1 : 0.035;
        (this.presence += (this.presenceTarget - this.presence) * (1 - Math.pow(1 - F, N))),
          this.presence < 0.001 && this.presenceTarget === 0 && (this.presence = 0);
        const tt = this.presenceTarget > this.flowing ? 0.05 : 0.018;
        (this.flowing += (this.presenceTarget - this.flowing) * (1 - Math.pow(1 - tt, N))),
          this.flowing < 0.002 && this.presenceTarget === 0 && (this.flowing = 0),
          this.introT < 1 && (this.introT = Math.min(1, this.introT + Math.min(G, 100) / 3200));
        const w = this.presence > 0.001 || this.burst > 0;
        if (((this.settle = w ? 0 : this.settle + N), !(w || this.settle < SETTLE_FRAMES || this.introT < 1))) {
          if ((this.flowDirty && this.clearFlow(), this.stateDirty && this.clearState(), !D)) {
            this.draw(), this.firstFrame(), this.stop();
            return;
          }
          if (((this.raf = requestAnimationFrame(this.frame)), this.skip++ % 2)) return;
          (this.time += (N * 2 * K.speed) / 60), this.draw(), this.firstFrame();
          return;
        }
        (this.raf = requestAnimationFrame(this.frame)),
          (this.time += (N * K.speed * (1 + this.flowing * 1.6)) / 60),
          this.stepFlow(N),
          this.stepSim(N),
          (this.burst = Math.max(0, this.burst - 0.1 * N)),
          this.draw(),
          this.firstFrame();
      }),
      (this.canvas = o),
      (this.opts = h),
      (this.onFail = s);
    const _ = o.getContext("webgl2", {
      alpha: !0,
      antialias: !1,
      depth: !1,
      stencil: !1,
      premultipliedAlpha: !0,
      preserveDrawingBuffer: !1,
      powerPreference: "high-performance",
    });
    if (!_) throw new Error("WebGL2 unavailable");
    this.gl = _;
    const x = navigator;
    (this.lowPower =
      (x.hardwareConcurrency || 8) <= 4 ||
      (x.deviceMemory || 8) <= 4 ||
      window.matchMedia("(pointer: coarse)").matches),
      this.initGL(),
      o.addEventListener("webglcontextlost", this.onLost, !1);
  }
  initGL() {
    const o = this.gl,
      h = o.getExtension("EXT_color_buffer_float"),
      s = h ? null : o.getExtension("EXT_color_buffer_half_float");
    if (!h && !s) throw new Error("No float render targets");
    (this.stateFormat = { internal: o.RGBA16F, type: o.HALF_FLOAT }),
      (this.pFlow = createProgram(o, QUAD_VS, FLOW_FS)),
      (this.pSim = createProgram(o, QUAD_VS, SIM_FS)),
      (this.pPart = createProgram(o, PARTICLE_VS, PARTICLE_FS)),
      (this.pImg = createProgram(o, QUAD_VS, IMAGE_FS)),
      (this.vao = o.createVertexArray()),
      o.bindVertexArray(this.vao),
      (this.quad = o.createBuffer()),
      o.bindBuffer(o.ARRAY_BUFFER, this.quad),
      o.bufferData(o.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), o.STATIC_DRAW),
      o.enableVertexAttribArray(0),
      o.vertexAttribPointer(0, 2, o.FLOAT, !1, 0, 0);
    for (let _ = 0; _ < 2; _++) {
      const x = this.makeTex(o.RGBA8, this.flowW, this.flowH, o.RGBA, o.UNSIGNED_BYTE, null, o.LINEAR),
        R = o.createFramebuffer();
      o.bindFramebuffer(o.FRAMEBUFFER, R),
        o.framebufferTexture2D(o.FRAMEBUFFER, o.COLOR_ATTACHMENT0, o.TEXTURE_2D, x, 0),
        this.flowTex.push(x),
        this.flowFb.push(R);
    }
    this.clearFlow();
  }
  makeTex(o, h, s, _, x, R, G) {
    const N = this.gl,
      K = N.createTexture();
    return (
      N.bindTexture(N.TEXTURE_2D, K),
      N.pixelStorei(N.UNPACK_ALIGNMENT, 1),
      N.texImage2D(N.TEXTURE_2D, 0, o, h, s, 0, _, x, R),
      N.texParameteri(N.TEXTURE_2D, N.TEXTURE_MIN_FILTER, G),
      N.texParameteri(N.TEXTURE_2D, N.TEXTURE_MAG_FILTER, G),
      N.texParameteri(N.TEXTURE_2D, N.TEXTURE_WRAP_S, N.CLAMP_TO_EDGE),
      N.texParameteri(N.TEXTURE_2D, N.TEXTURE_WRAP_T, N.CLAMP_TO_EDGE),
      K
    );
  }
  clearFlow() {
    const o = this.gl;
    for (const h of this.flowFb)
      o.bindFramebuffer(o.FRAMEBUFFER, h),
        o.viewport(0, 0, this.flowW, this.flowH),
        o.clearColor(128 / 255, 128 / 255, 0, 1),
        o.clear(o.COLOR_BUFFER_BIT);
    o.bindFramebuffer(o.FRAMEBUFFER, null), (this.flowDirty = !1);
  }
  clearState() {
    const o = this.gl;
    for (const h of this.stateFb)
      o.bindFramebuffer(o.FRAMEBUFFER, h),
        o.viewport(0, 0, STATE_TEX_W, this.texH),
        o.clearColor(0, 0, 0, 0),
        o.clear(o.COLOR_BUFFER_BIT);
    o.bindFramebuffer(o.FRAMEBUFFER, null), (this.stateDirty = !1);
  }
  setImage(o) {
    if (this.destroyed || this.lost) return;
    const h = this.gl;
    (this.lastImage = o),
      (this.aspect = o.naturalWidth / o.naturalHeight || 2.5),
      (this.flowH = Math.max(32, Math.round(this.flowW / this.aspect)));
    for (const s of this.flowTex)
      h.bindTexture(h.TEXTURE_2D, s),
        h.texImage2D(h.TEXTURE_2D, 0, h.RGBA8, this.flowW, this.flowH, 0, h.RGBA, h.UNSIGNED_BYTE, null);
    this.clearFlow(),
      this.imgTex && h.deleteTexture(this.imgTex),
      (this.imgTex = h.createTexture()),
      h.bindTexture(h.TEXTURE_2D, this.imgTex),
      h.pixelStorei(h.UNPACK_FLIP_Y_WEBGL, !1),
      h.texImage2D(h.TEXTURE_2D, 0, h.RGBA, h.RGBA, h.UNSIGNED_BYTE, o),
      h.generateMipmap(h.TEXTURE_2D),
      h.texParameteri(h.TEXTURE_2D, h.TEXTURE_MIN_FILTER, h.LINEAR_MIPMAP_LINEAR),
      h.texParameteri(h.TEXTURE_2D, h.TEXTURE_MAG_FILTER, h.LINEAR),
      h.texParameteri(h.TEXTURE_2D, h.TEXTURE_WRAP_S, h.CLAMP_TO_EDGE),
      h.texParameteri(h.TEXTURE_2D, h.TEXTURE_WRAP_T, h.CLAMP_TO_EDGE),
      this.buildParticles(o),
      (this.ready = !0),
      this.opts.intro && !this.opts.reducedMotion && this.startIntro(),
      this.wake();
  }
  buildParticles(o) {
    const h = this.gl,
      s = this.lowPower ? 640 : 1e3,
      _ = Math.round(s * Math.sqrt(clamp(this.opts.density, 0.15, 2))),
      x = Math.max(1, Math.round(_ / this.aspect));
    this.gridW = _;
    let R;
    try {
      const F = document.createElement("canvas");
      (F.width = _), (F.height = x);
      const tt = F.getContext("2d", { willReadFrequently: !0 });
      (tt.imageSmoothingQuality = "high"),
        tt.drawImage(o, 0, 0, _, x),
        (R = tt.getImageData(0, 0, _, x).data);
    } catch {
      throw new Error("Image is not CORS-readable");
    }
    let G = 0;
    for (let F = 3; F < R.length; F += 28) R[F] < 250 && G++;
    if (((this.alphaMode = G > R.length / 28 / 20), this.opts.imgBg)) this.imgBg = this.opts.imgBg;
    else if (!this.alphaMode) {
      let F = 0,
        tt = 0,
        w = 0,
        Ut = 0;
      const At = Math.max(2, Math.round(_ * 0.012));
      for (const [vt, at] of [
        [0, 0],
        [_ - At, 0],
        [0, x - At],
        [_ - At, x - At],
      ])
        for (let qt = at; qt < at + At; qt++)
          for (let xt = vt; xt < vt + At; xt++) {
            const $ = (qt * _ + xt) * 4;
            (F += R[$]), (tt += R[$ + 1]), (w += R[$ + 2]), Ut++;
          }
      this.imgBg = [F / Ut / 255, tt / Ut / 255, w / Ut / 255];
    }
    const N = [],
      K = [];
    for (let F = 0; F < x; F++)
      for (let tt = 0; tt < _; tt++) {
        const w = (F * _ + tt) * 4,
          Ut = R[w] / 255,
          At = R[w + 1] / 255,
          vt = R[w + 2] / 255;
        if (this.alphaMode) {
          const xt = R[w + 3] / 255;
          if (xt < 0.12 || (xt < 0.5 && Math.random() > xt * 2)) continue;
        } else {
          const xt = Math.hypot(Ut - this.imgBg[0], At - this.imgBg[1], vt - this.imgBg[2]);
          if (xt < 0.03 || (xt < 0.07 && Math.random() > (xt - 0.03) / 0.04)) continue;
        }
        const at = (Math.random() - 0.5) * 0.6,
          qt = (Math.random() - 0.5) * 0.6;
        N.push((tt + 0.5 + at) / _, (F + 0.5 + qt) / x, Math.random(), 0.85 + Math.random() * 0.3),
          K.push(R[w], R[w + 1], R[w + 2], 255);
      }
    const D = N.length / 4;
    (this.n = D), (this.texH = Math.max(1, Math.ceil(D / STATE_TEX_W)));
    const S = STATE_TEX_W * this.texH,
      H = new Float32Array(S * 4);
    H.set(N);
    const _t = new Uint8Array(S * 4);
    _t.set(K);
    for (const F of [this.homeTex, this.colTex, ...this.stateTex]) F && h.deleteTexture(F);
    for (const F of this.stateFb) h.deleteFramebuffer(F);
    (this.stateTex = []),
      (this.stateFb = []),
      (this.homeTex = this.makeTex(h.RGBA32F, STATE_TEX_W, this.texH, h.RGBA, h.FLOAT, H, h.NEAREST)),
      (this.colTex = this.makeTex(h.RGBA8, STATE_TEX_W, this.texH, h.RGBA, h.UNSIGNED_BYTE, _t, h.NEAREST));
    for (let F = 0; F < 2; F++) {
      const tt = this.makeTex(
          this.stateFormat.internal,
          STATE_TEX_W,
          this.texH,
          h.RGBA,
          this.stateFormat.type,
          null,
          h.NEAREST,
        ),
        w = h.createFramebuffer();
      if (
        (h.bindFramebuffer(h.FRAMEBUFFER, w),
        h.framebufferTexture2D(h.FRAMEBUFFER, h.COLOR_ATTACHMENT0, h.TEXTURE_2D, tt, 0),
        h.checkFramebufferStatus(h.FRAMEBUFFER) !== h.FRAMEBUFFER_COMPLETE)
      )
        throw new Error("Float FBO incomplete");
      this.stateTex.push(tt), this.stateFb.push(w);
    }
    h.bindFramebuffer(h.FRAMEBUFFER, null), (this.homeData = H), this.clearState();
  }
  startIntro() {
    const o = this.gl,
      h = this.homeData;
    if (!h) return;
    const s = this.rect.w > 10 ? this.rect.w : 1300,
      _ = STATE_TEX_W * this.texH,
      x = new Float32Array(_ * 4);
    for (let R = 0; R < this.n; R++) {
      const G = h[R * 4] - 0.5,
        N = h[R * 4 + 1] - 0.5,
        K = h[R * 4 + 2],
        D = (Math.random() - 0.5) * 0.9 + (G > 0 ? 0 : Math.PI),
        S = s * (0.04 + Math.random() * 0.14) * (0.5 + Math.abs(G) * 1.4);
      (x[R * 4] = Math.cos(D) * S + G * s * 0.25 * K),
        (x[R * 4 + 1] = Math.sin(D) * S * 0.55 + N * 120 * (Math.random() - 0.5));
    }
    o.bindTexture(o.TEXTURE_2D, this.stateTex[this.stateIdx]),
      o.texImage2D(o.TEXTURE_2D, 0, this.stateFormat.internal, STATE_TEX_W, this.texH, 0, o.RGBA, o.FLOAT, x),
      (this.introT = 0),
      (this.stateDirty = !0);
  }
  setOptions(o) {
    const h =
      (o.density !== void 0 && o.density !== this.opts.density) ||
      (o.imgBg !== void 0 && String(o.imgBg) !== String(this.opts.imgBg));
    if (((this.opts = { ...this.opts, ...o }), h && this.ready && this.lastImage))
      try {
        this.buildParticles(this.lastImage);
      } catch {
        this.onFail();
      }
    this.wake();
  }
  resize(o, h, s) {
    (this.cssW = Math.max(1, o)), (this.cssH = Math.max(1, h)), (this.rect = s);
    let _ = Math.min(window.devicePixelRatio || 1, this.lowPower ? 1.5 : 2);
    this.cssW * this.cssH * _ * _ > 45e5 && (_ = Math.max(1, Math.sqrt(45e5 / (this.cssW * this.cssH))));
    const R = Math.round(this.cssW * _),
      G = Math.round(this.cssH * _);
    (this.canvas.width !== R || this.canvas.height !== G) &&
      ((this.canvas.width = R), (this.canvas.height = G)),
      this.draw();
  }
  setVisible(o) {
    (this.visible = o), o ? this.wake() : this.stop();
  }
  pointer(o, h, s, _) {
    if (!this.opts.interactive || this.lost) return;
    const x = performance.now();
    if (o === "leave" || o === "up") {
      _ ? (this.touchRelease = x + 260) : (this.presenceTarget = 0), this.wake();
      return;
    }
    (this.mouse.x < -9e3 || o === "down") &&
      ((this.mouse.x = this.prevMouse.x = h), (this.mouse.y = this.prevMouse.y = s)),
      (this.target.x = h),
      (this.target.y = s),
      (this.presenceTarget = 1),
      (this.lastInput = x),
      (this.settle = 0),
      o === "down" && ((this.burst = 1), (this.touchRelease = 0)),
      this.wake();
  }
  wake() {
    if (!this.ready || this.running || !this.visible || this.destroyed || this.lost) {
      this.ready && !this.running && this.draw();
      return;
    }
    (this.running = !0), (this.last = performance.now()), (this.raf = requestAnimationFrame(this.frame));
  }
  stop() {
    (this.running = !1), cancelAnimationFrame(this.raf);
  }
  radiusPx() {
    return this.opts.radius * clamp(this.rect.w / 1400, 0.6, 1.25);
  }
  strength() {
    return this.opts.strength * (this.opts.reducedMotion ? 0.5 : 1);
  }
  firstFrame() {
    if (this.onFirstFrame) {
      const o = this.onFirstFrame;
      (this.onFirstFrame = null), requestAnimationFrame(() => o());
    }
  }
  stepFlow(o) {
    const h = this.gl,
      s = this.pFlow,
      _ = 1 - this.flowIdx;
    h.bindFramebuffer(h.FRAMEBUFFER, this.flowFb[_]),
      h.viewport(0, 0, this.flowW, this.flowH),
      h.disable(h.BLEND),
      h.useProgram(s.p),
      h.bindVertexArray(this.vao),
      h.activeTexture(h.TEXTURE0),
      h.bindTexture(h.TEXTURE_2D, this.flowTex[this.flowIdx]),
      h.uniform1i(s.u.uPrev, 0),
      h.uniform2f(s.u.uTexel, 1 / this.flowW, 1 / this.flowH);
    const x = this.rect;
    h.uniform2f(s.u.uMouse, (this.mouse.x - x.x) / x.w, (this.mouse.y - x.y) / x.h),
      h.uniform2f(s.u.uVel, clamp(this.mvel.x / 40, -1, 1), clamp(this.mvel.y / 40, -1, 1)),
      h.uniform2f(s.u.uSize, x.w, x.h),
      h.uniform1f(s.u.uRadius, this.radiusPx()),
      h.uniform1f(s.u.uPresence, Math.max(this.presence, this.burst)),
      h.uniform1f(s.u.uDecay, Math.pow(0.968, o)),
      h.drawArrays(h.TRIANGLE_STRIP, 0, 4),
      (this.flowIdx = _),
      (this.flowDirty = !0);
  }
  stepSim(o) {
    if (!this.n) return;
    const h = this.gl,
      s = this.pSim,
      _ = this.opts,
      x = this.rect,
      R = 1 - this.stateIdx,
      G = _.reducedMotion || !_.intro ? 0 : Math.pow(1 - this.introT, 2);
    h.bindFramebuffer(h.FRAMEBUFFER, this.stateFb[R]),
      h.viewport(0, 0, STATE_TEX_W, this.texH),
      h.disable(h.BLEND),
      h.useProgram(s.p),
      h.bindVertexArray(this.vao),
      h.activeTexture(h.TEXTURE0),
      h.bindTexture(h.TEXTURE_2D, this.stateTex[this.stateIdx]),
      h.uniform1i(s.u.uState, 0),
      h.activeTexture(h.TEXTURE1),
      h.bindTexture(h.TEXTURE_2D, this.homeTex),
      h.uniform1i(s.u.uHome, 1),
      h.activeTexture(h.TEXTURE2),
      h.bindTexture(h.TEXTURE_2D, this.flowTex[this.flowIdx]),
      h.uniform1i(s.u.uFlow, 2),
      h.uniform2f(s.u.uSize, x.w, x.h),
      h.uniform2f(s.u.uMouse, this.mouse.x - x.x, this.mouse.y - x.y),
      h.uniform1f(s.u.uRadius, this.radiusPx()),
      h.uniform1f(s.u.uPresence, Math.max(this.presence, this.burst)),
      h.uniform1f(s.u.uStrength, this.strength() * (1 + this.burst * 1.5)),
      h.uniform1f(s.u.uSpread, clamp(_.spread, 0, 2)),
      h.uniform1f(s.u.uTurb, clamp(_.turbulence, 0, 2) * (_.reducedMotion ? 0.3 : 1)),
      h.uniform1f(s.u.uReturn, clamp(_.returnSpeed, 0, 2)),
      h.uniform1f(s.u.uTime, this.time),
      h.uniform1f(s.u.uDt, o),
      h.uniform1f(s.u.uIntro, G),
      h.drawArrays(h.TRIANGLE_STRIP, 0, 4),
      (this.stateIdx = R),
      (this.stateDirty = !0);
  }
  draw() {
    if (!this.ready || this.lost || this.destroyed) return;
    const o = this.gl,
      h = this.opts,
      s = this.rect,
      _ = h.reducedMotion,
      x = _ || !h.intro ? 1 : Math.pow(this.introT, 2.2),
      G =
        (h.ambient && !_ ? clamp(h.ambientIntensity, 0, 2) : 0) +
        this.flowing * 0.85 * clamp(h.liquid, 0, 2) * (_ ? 0.3 : 1),
      N = h.restStyle === "particles" ? 1 : 0,
      [K, D, S] = h.bg,
      [H, _t, F] = this.imgBg,
      tt = this.canvas.width / this.cssW,
      w = Math.max(this.presence, this.burst),
      Ut = this.mouse.x - s.x,
      At = this.mouse.y - s.y;
    o.bindFramebuffer(o.FRAMEBUFFER, null),
      o.viewport(0, 0, this.canvas.width, this.canvas.height),
      o.bindVertexArray(this.vao),
      o.disable(o.BLEND);
    const vt = this.pImg;
    if (
      (o.useProgram(vt.p),
      o.activeTexture(o.TEXTURE0),
      o.bindTexture(o.TEXTURE_2D, this.imgTex),
      o.uniform1i(vt.u.uImg, 0),
      o.activeTexture(o.TEXTURE1),
      o.bindTexture(o.TEXTURE_2D, this.flowTex[this.flowIdx]),
      o.uniform1i(vt.u.uFlow, 1),
      o.uniform2f(vt.u.uRes, this.cssW, this.cssH),
      o.uniform1f(vt.u.uDpr, tt),
      o.uniform4f(vt.u.uRect, s.x, s.y, s.w, s.h),
      o.uniform3f(vt.u.uBg, K, D, S),
      o.uniform3f(vt.u.uImgBg, H, _t, F),
      o.uniform2f(vt.u.uMouse, Ut, At),
      o.uniform1f(vt.u.uPresence, w),
      o.uniform1f(vt.u.uRadius, this.radiusPx()),
      o.uniform1f(vt.u.uStrength, this.strength()),
      o.uniform1f(vt.u.uTime, this.time),
      o.uniform1f(vt.u.uAmbient, G),
      o.uniform1f(vt.u.uIntroVis, x),
      o.uniform1f(vt.u.uRest, N),
      o.uniform1f(vt.u.uAlphaMode, this.alphaMode ? 1 : 0),
      o.drawArrays(o.TRIANGLE_STRIP, 0, 4),
      !this.n || (!this.stateDirty && N === 0 && x >= 1))
    )
      return;
    const at = this.pPart;
    o.useProgram(at.p),
      o.enable(o.BLEND),
      o.blendFunc(o.ONE, o.ONE_MINUS_SRC_ALPHA),
      o.activeTexture(o.TEXTURE0),
      o.bindTexture(o.TEXTURE_2D, this.stateTex[this.stateIdx]),
      o.uniform1i(at.u.uState, 0),
      o.activeTexture(o.TEXTURE1),
      o.bindTexture(o.TEXTURE_2D, this.homeTex),
      o.uniform1i(at.u.uHome, 1),
      o.activeTexture(o.TEXTURE2),
      o.bindTexture(o.TEXTURE_2D, this.colTex),
      o.uniform1i(at.u.uCol, 2),
      o.activeTexture(o.TEXTURE3),
      o.bindTexture(o.TEXTURE_2D, this.flowTex[this.flowIdx]),
      o.uniform1i(at.u.uFlow, 3),
      o.uniform1i(at.u.uTexW, STATE_TEX_W),
      o.uniform4f(at.u.uRect, s.x, s.y, s.w, s.h),
      o.uniform2f(at.u.uRes, this.cssW, this.cssH),
      o.uniform2f(at.u.uMouse, Ut, At),
      o.uniform1f(at.u.uRadius, this.radiusPx()),
      o.uniform1f(at.u.uPresence, w);
    const qt = s.w / this.gridW;
    o.uniform1f(at.u.uDot, qt * 1.45 * clamp(h.dotSize, 0.2, 3)),
      o.uniform1f(at.u.uTime, this.time),
      o.uniform1f(at.u.uAmbient, G),
      o.uniform1f(at.u.uStreak, (h.streaks ? 1 : 0) * (0.45 + 0.55 * x)),
      o.uniform1f(at.u.uIntroVis, x),
      o.uniform1f(at.u.uRest, N),
      o.uniform1f(at.u.uDpr, tt),
      this.alphaMode
        ? o.uniform3f(at.u.uTint, 1, 1, 1)
        : o.uniform3f(at.u.uTint, K / Math.max(H, 0.001), D / Math.max(_t, 0.001), S / Math.max(F, 0.001)),
      o.drawArraysInstanced(o.TRIANGLE_STRIP, 0, 4, this.n),
      o.disable(o.BLEND);
  }
  destroy() {
    var h;
    (this.destroyed = !0), this.stop(), this.canvas.removeEventListener("webglcontextlost", this.onLost);
    const o = this.gl;
    if (!this.lost) {
      o.deleteBuffer(this.quad), o.deleteVertexArray(this.vao);
      for (const s of [...this.flowTex, ...this.stateTex, this.imgTex, this.homeTex, this.colTex])
        s && o.deleteTexture(s);
      for (const s of [...this.flowFb, ...this.stateFb]) o.deleteFramebuffer(s);
      for (const s of [this.pFlow, this.pSim, this.pPart, this.pImg]) s && o.deleteProgram(s.p);
    }
    (h = o.getExtension("WEBGL_lose_context")) == null || h.loseContext();
  }
}


// ---------------------------------------------------------------------------
// Retake wrapper (no framework)

const SCULPTURE_IMAGES = {
  light: { wide: "./assets/forma/sculpture-light-wide.webp", tall: "./assets/forma/sculpture-light-tall.webp" },
  dark: { wide: "./assets/forma/sculpture-dark-wide.webp", tall: "./assets/forma/sculpture-dark-tall.webp" }
};

const DEFAULT_ENGINE_OPTIONS = {
  interactive: true,
  strength: 1,
  radius: 220,
  density: 1,
  dotSize: 1,
  spread: 1,
  turbulence: 1,
  liquid: 1,
  streaks: true,
  restStyle: "image",
  returnSpeed: 1,
  ambient: true,
  ambientIntensity: 0.6,
  speed: 1,
  intro: true,
  imgBg: null
};

export function parseCssColor(value, fallback = [0.92, 0.91, 0.9]) {
  const text = String(value || "").trim();
  let match = /^#?([0-9a-f]{3})$/i.exec(text);
  if (match) return match[1].split("").map((digit) => parseInt(digit + digit, 16) / 255);
  match = /^#?([0-9a-f]{6})([0-9a-f]{2})?$/i.exec(text);
  if (match) {
    const number = parseInt(match[1], 16);
    return [(number >> 16 & 255) / 255, (number >> 8 & 255) / 255, (number & 255) / 255];
  }
  match = /^rgba?\(\s*([\d.]+)[,\s]+([\d.]+)[,\s]+([\d.]+)/i.exec(text);
  return match ? [+match[1] / 255, +match[2] / 255, +match[3] / 255] : fallback;
}

/** Wide image for landscape boxes, tall image for portrait ones. */
export function pickSculptureImage(theme, width, height) {
  const set = SCULPTURE_IMAGES[theme === "light" ? "light" : "dark"];
  return width / Math.max(height, 1) >= 1.15 ? set.wide : set.tall;
}

/** Fit the sculpture (aspect = w/h) inside a box, centered, like object-fit: contain. */
export function fitSculptureRect(boxWidth, boxHeight, aspect, scale = 1) {
  let width = boxWidth * scale;
  let height = width / aspect;
  if (height > boxHeight * scale) {
    height = boxHeight * scale;
    width = height * aspect;
  }
  return { x: (boxWidth - width) / 2, y: (boxHeight - height) / 2, w: width, h: height };
}

/**
 * Particle density for a sculpture drawn `width` CSS px wide. The engine's
 * grid is 1000 particles across at density 1 — sized for a full-screen hero
 * about 1300 px wide (~0.77 particles per px). Smaller boxes get the same
 * on-screen density with far fewer particles.
 */
export function densityForWidth(width, particlesPerPx = 0.8) {
  const grid = Math.max(1, width) * particlesPerPx;
  return Math.min(1, Math.max(0.15, (grid / 1000) ** 2));
}

/**
 * Mount the sculpture into `host` (a positioned element). Returns a controller,
 * or null when WebGL2 is unavailable — the caller keeps its static fallback.
 */
export function mountSculpture(host, {
  theme = () => document.documentElement.dataset.theme === "light" ? "light" : "dark",
  background = () => getComputedStyle(document.documentElement).getPropertyValue("--bg"),
  scale = 1,
  autopilot = false,
  options = {}
} = {}) {
  if (!host) return null;
  const canvas = document.createElement("canvas");
  canvas.className = "sculpture-canvas";
  canvas.setAttribute("aria-hidden", "true");
  host.append(canvas);

  const reducedMotion = globalThis.matchMedia?.("(prefers-reduced-motion: reduce)");
  let failed = false;
  let aspect = 1983 / 793;
  let density = densityForWidth(fitSculptureRect(host.clientWidth || 600, host.clientHeight || 300, aspect, scale).w);
  const fail = () => {
    failed = true;
    host.dataset.gl = "off";
  };
  const engineOptions = () => ({
    ...DEFAULT_ENGINE_OPTIONS,
    ...options,
    reducedMotion: Boolean(reducedMotion?.matches),
    bg: parseCssColor(background()),
    density
  });

  let engine;
  try {
    engine = new SculptureEngine(canvas, engineOptions(), fail);
  } catch (error) {
    console.info("Heykel animasyonu başlatılamadı; sabit görsel kullanılıyor.", error);
    canvas.remove();
    return null;
  }

  let currentSrc = "";
  let imageToken = 0;

  function layout() {
    const width = host.clientWidth;
    const height = host.clientHeight;
    if (width < 2 || height < 2) return;
    const rect = fitSculptureRect(width, height, aspect, scale);
    const wanted = densityForWidth(rect.w);
    // Rebuilding the particles is costly: only for real size changes.
    if (Math.abs(wanted - density) / density > 0.25) {
      density = wanted;
      engine.setOptions({ density });
    }
    engine.resize(width, height, rect);
  }

  function loadImage() {
    const src = pickSculptureImage(theme(), host.clientWidth, host.clientHeight);
    if (src === currentSrc) return;
    currentSrc = src;
    const token = ++imageToken;
    const image = new Image();
    image.decoding = "async";
    image.onload = () => {
      if (token !== imageToken || failed) return;
      aspect = image.naturalWidth / image.naturalHeight || aspect;
      layout();
      try {
        engine.onFirstFrame = () => {
          if (token === imageToken) host.dataset.gl = "on";
        };
        engine.setImage(image);
        layout();
      } catch (error) {
        console.info("Heykel görseli parçacıklara dönüştürülemedi.", error);
        fail();
      }
    };
    image.src = src;
  }

  const local = (event) => {
    const rect = host.getBoundingClientRect();
    return [event.clientX - rect.left, event.clientY - rect.top];
  };
  let lastHuman = -Infinity;
  const onMove = (event) => {
    const touch = event.pointerType === "touch";
    if (touch && event.buttons === 0 && event.pressure === 0) return;
    lastHuman = performance.now();
    engine.pointer("move", ...local(event), touch);
    host.dataset.touched = "1";
  };
  const onDown = (event) => {
    lastHuman = performance.now();
    engine.pointer("down", ...local(event), event.pointerType === "touch");
    host.dataset.touched = "1";
  };
  const onUp = (event) => engine.pointer("up", ...local(event), event.pointerType === "touch");
  const onLeave = (event) => engine.pointer("leave", ...local(event), event.pointerType === "touch");
  host.addEventListener("pointermove", onMove, { passive: true });
  host.addEventListener("pointerdown", onDown, { passive: true });
  host.addEventListener("pointerup", onUp, { passive: true });
  host.addEventListener("pointercancel", onUp, { passive: true });
  host.addEventListener("pointerleave", onLeave, { passive: true });

  const resizeObserver = new ResizeObserver(() => {
    loadImage();
    layout();
  });
  resizeObserver.observe(host);
  let visible = true;
  const intersection = new IntersectionObserver((entries) => {
    visible = entries.some((entry) => entry.isIntersecting);
    engine.setVisible(visible && !document.hidden);
  });
  intersection.observe(host);
  const onVisibility = () => engine.setVisible(visible && !document.hidden);
  document.addEventListener("visibilitychange", onVisibility);
  const onMotion = () => engine.setOptions({ reducedMotion: Boolean(reducedMotion?.matches) });
  reducedMotion?.addEventListener?.("change", onMotion);

  // Autopilot: while nobody touches it, an invisible hand drifts through the
  // sculpture so it keeps flowing on its own (used while a photo is processed).
  let pilot = autopilot;
  let pilotFrame = 0;
  let pilotStart = performance.now();
  function steer(now) {
    pilotFrame = 0;
    if (!pilot || failed) return;
    const idle = now - lastHuman > 2500;
    if (idle && visible && !document.hidden && !reducedMotion?.matches) {
      const t = (now - pilotStart) / 1000;
      const width = host.clientWidth;
      const height = host.clientHeight;
      const x = width * (0.5 + 0.34 * Math.sin(t * 0.55) * Math.cos(t * 0.13));
      const y = height * (0.5 + 0.22 * Math.sin(t * 0.83 + 1.1));
      engine.pointer("move", x, y, false);
    }
    pilotFrame = requestAnimationFrame(steer);
  }
  function setAutopilot(on) {
    pilot = Boolean(on);
    if (pilot && !pilotFrame) {
      pilotStart = performance.now();
      pilotFrame = requestAnimationFrame(steer);
    } else if (!pilot) {
      if (pilotFrame) cancelAnimationFrame(pilotFrame);
      pilotFrame = 0;
      engine.pointer("leave", 0, 0, false);
    }
  }

  loadImage();
  layout();
  setAutopilot(pilot);

  return {
    /** Re-read theme colours (call after a theme switch). */
    refreshTheme() {
      engine.setOptions({ bg: parseCssColor(background()) });
      loadImage();
    },
    setAutopilot,
    setOptions(next) {
      Object.assign(options, next);
      engine.setOptions(next);
    },
    destroy() {
      setAutopilot(false);
      resizeObserver.disconnect();
      intersection.disconnect();
      document.removeEventListener("visibilitychange", onVisibility);
      reducedMotion?.removeEventListener?.("change", onMotion);
      host.removeEventListener("pointermove", onMove);
      host.removeEventListener("pointerdown", onDown);
      host.removeEventListener("pointerup", onUp);
      host.removeEventListener("pointercancel", onUp);
      host.removeEventListener("pointerleave", onLeave);
      engine.destroy();
      canvas.remove();
      delete host.dataset.gl;
    }
  };
}
