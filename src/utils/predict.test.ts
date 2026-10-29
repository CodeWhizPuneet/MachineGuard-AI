import test from 'node:test';
import assert from 'node:assert/strict';
import {
  validateMachineInput,
  computeMachineFailureRisk,
  diagnoseMachine,
  processBatchCsv,
  BASELINE_REFERENCE,
} from './predict';

test('validateMachineInput accepts valid physical parameters', () => {
  const result = validateMachineInput({
    type: 'M',
    airTemp: 300.0,
    procTemp: 310.0,
    rotSpeed: 1500,
    torque: 40.0,
    toolWear: 60,
  });
  assert.equal(result.isValid, true);
  assert.equal(Object.keys(result.errors).length, 0);
});

test('validateMachineInput flags out-of-range temperatures', () => {
  const result = validateMachineInput({
    type: 'M',
    airTemp: 450.0, // Invalid Kelvin
    procTemp: 310.0,
    rotSpeed: 1500,
    torque: 40.0,
    toolWear: 60,
  });
  assert.equal(result.isValid, false);
  assert.ok(result.errors.airTemp);
});

test('computeMachineFailureRisk calculates low probability for nominal conditions', () => {
  const prob = computeMachineFailureRisk(BASELINE_REFERENCE);
  assert.ok(prob < 0.15, `Expected low probability for nominal conditions, got ${prob}`);
});

test('computeMachineFailureRisk calculates elevated risk for extreme torque and wear', () => {
  const extremeInput = {
    type: 'L' as const,
    airTemp: 302.0,
    procTemp: 311.0,
    rotSpeed: 1250,
    torque: 75.0,  // Severe torque overload
    toolWear: 240, // Severe tool wear
  };
  const prob = computeMachineFailureRisk(extremeInput);
  assert.ok(prob > 0.70, `Expected high probability for extreme stress, got ${prob}`);
});

test('diagnoseMachine applies threshold and risk bands correctly', () => {
  const diag = diagnoseMachine(BASELINE_REFERENCE, 0.475);
  assert.equal(diag.predictedClass, 0);
  assert.equal(diag.riskBand, 'LOW_RISK');
  assert.ok(diag.contributions.length > 0);
});

test('processBatchCsv parses rows and excludes leakage columns', () => {
  const sampleCsv = `UDI,Product ID,Type,Air temperature [K],Process temperature [K],Rotational speed [rpm],Torque [Nm],Tool wear [min],Machine failure,TWF
1,L47180,L,298.1,308.6,1551,42.8,0,0,0
2,L47181,L,303.0,312.0,1200,68.0,225,1,1`;

  const { results, invalidRows, summary } = processBatchCsv(sampleCsv, 0.475);
  assert.equal(summary.valid, 2);
  assert.equal(invalidRows.length, 0);
  assert.equal(results[0].predictedClass, 0);
  // Second row with 68 Nm torque and 225 min wear should trigger high failure risk
  assert.equal(results[1].predictedClass, 1);
});
