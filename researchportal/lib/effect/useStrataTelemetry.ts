'use client';
import { useEffect, useRef, useState } from 'react';
import { initialTelemetry, startTelemetry } from './simulator';
export function useStrataTelemetry() { const [state, setState] = useState(initialTelemetry); const runtime = useRef<ReturnType<typeof startTelemetry> | null>(null); const [running, setRunning] = useState(true); const ef = useRef(80); useEffect(() => { if (!running)
    return; const r = startTelemetry(setState); r.setEf(ef.current); runtime.current = r; return () => { r.dispose(); runtime.current = null; }; }, [running]); return { state, running, setRunning, setEf: (v: number) => { ef.current = v; runtime.current?.setEf(v); setState(s => ({ ...s, efSearch: v })); } }; }
