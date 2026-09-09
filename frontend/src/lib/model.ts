"use client";
import { useEffect, useState } from "react";
import { API_URL } from "./config";

/**
 * What the backend is actually serving.
 *
 * The UI used to hard-code "ResNet-18 @ 224px" in the footer and the empty
 * state, so swapping the served checkpoint silently made the page lie. One
 * cached fetch, shared by every caller.
 */
export type ModelInfo = {
    architecture: string;
    input_size: number;
    temperature?: number;
    metrics?: Record<string, any>;
};

const PRETTY: Record<string, string> = {
    resnet18: "ResNet-18",
    resnet50: "ResNet-50",
    efficientnet_b0: "EfficientNet-B0",
};

export const prettyArch = (a?: string) =>
    a ? PRETTY[a] ?? a.replace(/_/g, "-") : "model";

export const modelLabel = (m?: ModelInfo | null) =>
    m ? `${prettyArch(m.architecture)} @ ${m.input_size}px` : "loading model…";

let cache: Promise<ModelInfo | null> | null = null;

export function useModelInfo(): ModelInfo | null {
    const [info, setInfo] = useState<ModelInfo | null>(null);

    useEffect(() => {
        if (!cache) {
            cache = fetch(`${API_URL}/model`)
                .then((r) => (r.ok ? r.json() : null))
                .catch(() => null);
        }
        let alive = true;
        cache.then((m) => alive && setInfo(m));
        return () => { alive = false; };
    }, []);

    return info;
}
