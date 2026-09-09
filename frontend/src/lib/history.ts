/**
 * Analysis history, kept in this browser.
 *
 * Overlays are ~120 KB of base64 each and localStorage caps near 5 MB, so only a
 * 160px thumbnail is kept. There are no accounts, so nothing here leaves the
 * device; supabase/schema.sql holds the table and policies if this ever needs
 * to sync.
 */

export type HistoryEntry = {
    id: string;
    at: number;
    fileName: string;
    thumb: string;
    risk: "HIGH" | "MEDIUM" | "LOW" | string;
    risk_probability: number;
    image_probability: number;
    confidence: number;
    predicted_class: string;
    iwgdf_category: number | null;
    attention_area_pct: number | null;
    focality: number | null;
    clinical_logit_shift: number;
    factors_supplied: string[];
    architecture: string;
    inference_ms: number;
};

const KEY = "cv-history";
const MAX_ENTRIES = 60;

const THUMB_PX = 160;

/** Downscale to a small JPEG data URL. Returns "" if the browser blocks canvas. */
export async function makeThumb(src: string, px = THUMB_PX): Promise<string> {
    try {
        const img = await new Promise<HTMLImageElement>((resolve, reject) => {
            const i = new Image();
            i.onload = () => resolve(i);
            i.onerror = reject;
            i.src = src;
        });
        const scale = Math.min(px / img.width, px / img.height, 1);
        const w = Math.max(1, Math.round(img.width * scale));
        const h = Math.max(1, Math.round(img.height * scale));
        const c = document.createElement("canvas");
        c.width = w;
        c.height = h;
        const ctx = c.getContext("2d");
        if (!ctx) return "";
        ctx.drawImage(img, 0, 0, w, h);
        return c.toDataURL("image/jpeg", 0.7);
    } catch {
        return "";
    }
}

export function toEntry(result: any, fileName: string, thumb: string): HistoryEntry {
    return {
        id: `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 7)}`,
        at: Date.now(),
        fileName,
        thumb,
        risk: result.risk,
        risk_probability: result.risk_probability,
        image_probability: result.image_probability,
        confidence: result.confidence,
        predicted_class: result.predicted_class,
        iwgdf_category: result.iwgdf?.category ?? null,
        attention_area_pct: result.attention?.attention_area_pct ?? null,
        focality: result.attention?.focality ?? null,
        clinical_logit_shift: result.clinical_logit_shift ?? 0,
        factors_supplied: result.factors_supplied ?? [],
        architecture: result.model?.architecture ?? "",
        inference_ms: result.inference_ms ?? 0,
    };
}

// --- localStorage backend -------------------------------------------------

function readLocal(): HistoryEntry[] {
    try {
        const raw = localStorage.getItem(KEY);
        const parsed = raw ? JSON.parse(raw) : [];
        return Array.isArray(parsed) ? parsed : [];
    } catch {
        return [];
    }
}

function writeLocal(entries: HistoryEntry[]) {
    try {
        localStorage.setItem(KEY, JSON.stringify(entries));
        return true;
    } catch {
        // Quota exceeded: drop the oldest half rather than lose the newest.
        try {
            localStorage.setItem(KEY, JSON.stringify(entries.slice(0, Math.floor(entries.length / 2))));
            return true;
        } catch {
            return false;
        }
    }
}

// --- Supabase backend (lazy) ----------------------------------------------


// --- public API -----------------------------------------------------------

export async function listHistory(): Promise<HistoryEntry[]> {
    return readLocal();
}

export async function addHistory(entry: HistoryEntry): Promise<HistoryEntry[]> {
    const next = [entry, ...readLocal()].slice(0, MAX_ENTRIES);
    writeLocal(next);
    return next;
}

export async function removeHistory(id: string): Promise<HistoryEntry[]> {
    const next = readLocal().filter((e) => e.id !== id);
    writeLocal(next);
    return next;
}

export async function clearHistory(): Promise<HistoryEntry[]> {
    try {
        localStorage.removeItem(KEY);
    } catch { }
    return [];
}

/** Aggregate counts for the summary strip. */
export function summarise(entries: HistoryEntry[]) {
    const n = entries.length;
    const by = { HIGH: 0, MEDIUM: 0, LOW: 0 } as Record<string, number>;
    let risk = 0;
    let ms = 0;
    for (const e of entries) {
        by[e.risk] = (by[e.risk] ?? 0) + 1;
        risk += e.risk_probability ?? 0;
        ms += e.inference_ms ?? 0;
    }
    return {
        n,
        high: by.HIGH ?? 0,
        medium: by.MEDIUM ?? 0,
        low: by.LOW ?? 0,
        meanRisk: n ? risk / n : 0,
        meanMs: n ? ms / n : 0,
        backend: "this browser",
    };
}

/** CSV of the whole history, for the export button. */
export function toCSV(entries: HistoryEntry[]): string {
    const cols: (keyof HistoryEntry)[] = [
        "at", "fileName", "risk", "risk_probability", "image_probability",
        "confidence", "predicted_class", "iwgdf_category", "attention_area_pct",
        "focality", "clinical_logit_shift", "architecture", "inference_ms",
    ];
    const esc = (v: any) => {
        const s = String(v ?? "");
        return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
    };
    const head = ["timestamp", ...cols.slice(1)].join(",");
    const rows = entries.map((e) =>
        [new Date(e.at).toISOString(), ...cols.slice(1).map((c) => esc(e[c]))].join(",")
    );
    return [head, ...rows].join("\n");
}
