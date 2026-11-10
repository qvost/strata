import { Effect, Fiber, Stream, Schedule, Ref } from 'effect';
export interface TelemetryState {
    efSearch: number;
    qps: number;
    latencyMs: number;
    l3Misses: number;
    dramKb: number;
    ipc: number;
    activeStage: 'COARSE_BQ' | 'FINE_SQ8' | 'IDLE';
    tick: number;
}
export const initialTelemetry: TelemetryState = { efSearch: 80, qps: 21600, latencyMs: .116, l3Misses: 536, dramKb: 405, ipc: 1.88, activeStage: 'IDLE', tick: 0 };
export const makeTelemetryStream = (efRef: Ref.Ref<number>) => Stream.fromSchedule(Schedule.spaced('50 millis')).pipe(Stream.mapEffect(tick => Effect.gen(function* () { const ef = yield* Ref.get(efRef); return { efSearch: ef, qps: Math.round(26000 - ef * 55 + Math.sin(tick * .13) * 100), l3Misses: Math.round(160 + ef * 4.7 + Math.sin(tick * .17) * 10), latencyMs: +(.038 + ef / 320 * .31).toFixed(3), dramKb: +(405 + Math.sin(tick * .1) * 4).toFixed(1), ipc: +(1.88 + Math.sin(tick * .11) * .04).toFixed(2), activeStage: tick % 20 < 13 ? 'COARSE_BQ' : 'FINE_SQ8', tick } as TelemetryState; })));
export function startTelemetry(onValue: (s: TelemetryState) => void) { const ref = Effect.runSync(Ref.make(80)); const fiber = Effect.runFork(Stream.runForEach(makeTelemetryStream(ref), s => Effect.sync(() => onValue(s)))); return { setEf: (ef: number) => Effect.runSync(Ref.set(ref, ef)), dispose: () => { Effect.runFork(Fiber.interrupt(fiber)); } }; }
export function startAnimation(draw: (time: number) => void) { const fiber = Effect.runFork(Effect.repeat(Effect.sync(() => draw(performance.now())), Schedule.spaced('32 millis'))); return () => { Effect.runFork(Fiber.interrupt(fiber)); }; }
