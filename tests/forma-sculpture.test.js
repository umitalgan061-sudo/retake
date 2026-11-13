import test from "node:test";
import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { densityForWidth, fitSculptureRect, placeWideSculpture, parseCssColor, pickSculptureImage } from "../src/forma-sculpture.js";

test("theme and box shape choose the sculpture image", () => {
  assert.match(pickSculptureImage("dark", 500, 300), /dark-wide/);
  assert.match(pickSculptureImage("light", 500, 300), /light-wide/);
  assert.match(pickSculptureImage("dark", 300, 400), /dark-tall/);
  assert.match(pickSculptureImage(undefined, 300, 0), /dark-wide/);
});

test("every sculpture image the page can ask for is in the repository", () => {
  for (const theme of ["light", "dark"]) {
    for (const [w, h] of [[500, 300], [300, 500]]) {
      const path = pickSculptureImage(theme, w, h).replace("./", "");
      assert.ok(existsSync(new URL("../" + path, import.meta.url)), path);
    }
  }
});

test("the sculpture is fitted and centered inside its box", () => {
  assert.deepEqual(fitSculptureRect(500, 300, 2.5), { x: 0, y: 50, w: 500, h: 200 });
  const tall = fitSculptureRect(300, 300, 0.75);
  assert.equal(tall.h, 300);
  assert.equal(tall.w, 225);
  assert.equal(tall.x, 37.5);
  const scaled = fitSculptureRect(500, 300, 2.5, 0.8);
  assert.equal(scaled.w, 400);
  assert.equal(scaled.x, 50);
});

test("CSS colours are parsed for the particle tint", () => {
  assert.deepEqual(parseCssColor("#ffffff"), [1, 1, 1]);
  assert.deepEqual(parseCssColor(" #000 "), [0, 0, 0]);
  assert.deepEqual(parseCssColor("rgb(255, 0, 0)"), [1, 0, 0]);
  assert.deepEqual(parseCssColor("nonsense", [0.5, 0.5, 0.5]), [0.5, 0.5, 0.5]);
});

test("the original author is credited on the page", () => {
  const html = readFileSync(new URL("../index.html", import.meta.url), "utf8");
  assert.match(html, /guglielmogiannattasio\.it/);
});

test("particle count follows the drawn size", () => {
  assert.equal(densityForWidth(1250), 1);
  assert.ok(Math.abs(densityForWidth(500) - 0.16) < 1e-9);
  assert.equal(densityForWidth(100), 0.15);
  assert.ok(densityForWidth(800) > densityForWidth(500));
});

test("wide hero spans the full width above the text", () => {
  const rect = placeWideSculpture(1900, 1000, 2.5, 800);
  assert.ok(Math.abs(rect.w - 1938) < 1e-9);
  assert.ok(rect.x < 0, "slightly past both edges");
  assert.ok(Math.abs(rect.y + rect.h / 2 - 400) < 1e-9, "centered in the free area");
  const short = placeWideSculpture(1900, 500, 2.5, 300);
  assert.ok(Math.abs(short.h - 330) < 1e-9, "never taller than 1.1x the area");
  const below = placeWideSculpture(1900, 1000, 2.5, 800, 64);
  assert.ok(Math.abs(below.y + below.h / 2 - 432) < 1e-9, "centered between the top margin and the text");
});
