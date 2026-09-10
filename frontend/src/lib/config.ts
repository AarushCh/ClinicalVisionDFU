// Single source of truth for anything the components share.

export const API_URL =
    process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:10000";

// basePath from next.config.mjs: a hard-coded "/samples/x.jpg" 404s on Pages.
export const BASE_PATH =
    process.env.NEXT_PUBLIC_BASE_PATH ?? "/ClinicalVisionDFU";

export const asset = (p: string) =>
    `${BASE_PATH}${p.startsWith("/") ? p : `/${p}`}`;

export type Preset = {
    id: string;
    label: string;
    caption: string;
    file: string;
    kind: "ulcer" | "healthy" | "ood";
    clinical: Record<string, string | boolean>;
    note: string;
};

// Demo cases from the group-aware TEST split. The whole-foot one is deliberate:
// it demonstrates the OOD failure measured in backend/ood_check.py.
export const PRESETS: Preset[] = [
    {
        id: "ulcer-1",
        label: "Plantar ulcer",
        caption: "Held-out test patch",
        file: "ulcer-1.jpg",
        kind: "ulcer",
        clinical: { age: "68", bmi: "31.2", diabetes_years: "14", hba1c: "8.6", neuropathy: true },
        note: "Long-standing diabetes with neuropathy. Expect HIGH risk and a tight heatmap on the wound bed.",
    },
    {
        id: "ulcer-2",
        label: "Deep ulceration",
        caption: "Held-out test patch",
        file: "ulcer-2.jpg",
        kind: "ulcer",
        clinical: { age: "74", bmi: "33.1", diabetes_years: "26", hba1c: "9.7", neuropathy: true, pad: true, smoker: true },
        note: "Neuropathy plus peripheral arterial disease — IWGDF category 2, the highest clinical modifier in the demo set.",
    },
    {
        id: "healthy-1",
        label: "Healthy skin",
        caption: "Held-out test patch",
        file: "healthy-1.jpg",
        kind: "healthy",
        clinical: { age: "48", bmi: "23.4", diabetes_years: "4", hba1c: "6.6" },
        note: "Well-controlled, low-risk profile. Expect LOW risk with diffuse, low-intensity attention.",
    },
    {
        id: "healthy-2",
        label: "Healthy — frail patient",
        caption: "Same image, harsher history",
        file: "healthy-2.jpg",
        kind: "healthy",
        clinical: { age: "81", bmi: "17.2", diabetes_years: "22", hba1c: "7.8", neuropathy: true },
        note: "Shows the fusion working: clean image, but underweight + neuropathy lifts fused risk well above the image probability alone.",
    },
    {
        id: "wholefoot",
        label: "Whole foot — known failure",
        caption: "Out of distribution",
        file: "wholefoot.jpg",
        kind: "ood",
        clinical: {},
        note: "Deliberately out of scope. The model was trained only on close crops and classifies ~100% of whole-foot photos as ulcer. This result carries no information — it is here to show the limitation, not to hide it.",
    },
];

// A stale backend shows up two ways: a 404 on an endpoint it never had, or a
// 422 demanding fields this build sends as optional. Same diagnosis, so say it.
const STALE =
    "The deployed backend is out of date and does not match this app.";

/**
 * One readable line from an axios failure.
 *
 * FastAPI's `detail` is a string for a raised HTTPException but an array of
 * {loc, msg} for a 422 — which used to reach the screen as a raw JSON blob.
 */
export function apiError(err: any, what = "the service"): string {
    if (err?.code === "ECONNABORTED")
        return `${what} timed out — it may be waking from a cold start. Try again.`;

    const res = err?.response;
    if (!res) return `Cannot reach ${what} at ${API_URL}.`;

    const detail = res.data?.detail;
    if (Array.isArray(detail)) {
        const fields = Array.from(
            new Set(detail.map((d: any) => d?.loc?.[1]).filter(Boolean))
        );
        return fields.length
            ? `${STALE} It is asking for ${fields.join(", ")}, which this version sends as optional.`
            : STALE;
    }
    // Before the string branch: FastAPI answers a missing route with "Not Found".
    if (res.status === 404) return STALE;
    if (typeof detail === "string" && detail) return detail;
    return `${what} returned error ${res.status}.`;
}
