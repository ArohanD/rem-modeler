import { describe, expect, it } from 'vitest';
import { capsProfile, planGrid, snapResolution, HARD_MAX_DIM, HARD_MAX_BBOX_KM } from '../src/caps';

describe('capsProfile', () => {
  it('scales the budget with device memory', () => {
    expect(capsProfile(8).budgetMB).toBe(512);
    expect(capsProfile(4).budgetMB).toBe(384);
    expect(capsProfile(2).budgetMB).toBe(192);
    expect(capsProfile(null).budgetMB).toBe(320);
  });

  it('never exceeds the hard dimension cap', () => {
    expect(capsProfile(8).maxDim).toBeLessThanOrEqual(HARD_MAX_DIM);
  });
});

describe('snapResolution', () => {
  it('snaps to power-of-two meters (COG overview levels)', () => {
    expect(snapResolution(0.4)).toBe(1);
    expect(snapResolution(1)).toBe(1);
    expect(snapResolution(1.1)).toBe(2);
    expect(snapResolution(3.5)).toBe(4);
    expect(snapResolution(9)).toBe(16);
  });
});

describe('planGrid', () => {
  const caps = capsProfile(8);

  it('keeps small areas at native 1 m resolution', () => {
    const plan = planGrid(3000, 2000, caps);
    expect(plan.rejected).toBe(false);
    expect(plan.res).toBe(1);
    expect(plan.width).toBe(3000);
    expect(plan.height).toBe(2000);
  });

  it('degrades resolution instead of exceeding the pixel budget', () => {
    const plan = planGrid(20000, 20000, caps);
    expect(plan.rejected).toBe(false);
    expect(plan.pixels).toBeLessThanOrEqual(caps.maxPixels);
    expect(plan.width).toBeLessThanOrEqual(caps.maxDim);
    expect(plan.height).toBeLessThanOrEqual(caps.maxDim);
    expect(plan.res).toBeGreaterThan(1);
  });

  it('stays within budget for every bbox size up to the hard cap', () => {
    for (let km = 1; km <= HARD_MAX_BBOX_KM; km++) {
      const plan = planGrid(km * 1000, km * 1000, caps);
      expect(plan.rejected).toBe(false);
      expect(plan.pixels).toBeLessThanOrEqual(caps.maxPixels);
      expect(plan.estimatedMB).toBeLessThanOrEqual(caps.budgetMB);
    }
  });

  it('rejects areas beyond the hard bbox cap', () => {
    const plan = planGrid((HARD_MAX_BBOX_KM + 1) * 1000, 5000, caps);
    expect(plan.rejected).toBe(true);
  });

  it('warns on large areas', () => {
    const plan = planGrid(20000, 20000, caps);
    expect(plan.warnings.length).toBeGreaterThan(0);
  });
});
