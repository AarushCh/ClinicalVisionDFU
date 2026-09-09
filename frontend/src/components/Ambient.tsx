"use client";
import { useEffect, useRef } from "react";

/**
 * Drifting colour fields, film grain and a cursor light.
 *
 * Stateless by design: the rAF loop writes transform on two elements and two
 * custom properties on a third -- compositor work only, no React render. It also
 * parks itself once the ring catches up, so a still cursor costs nothing.
 */

// Anything the ring should open up over.
const INTERACTIVE =
    'a,button,input,select,textarea,summary,label,[role="switch"],[role="tab"],[contenteditable="true"]';

export default function Ambient() {
    const ringRef = useRef<HTMLDivElement>(null);
    const dotRef = useRef<HTMLDivElement>(null);
    const spotRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        const fine = window.matchMedia("(hover: hover) and (pointer: fine)");
        const calm = window.matchMedia("(prefers-reduced-motion: reduce)");
        if (!fine.matches || calm.matches) return;

        const ring = ringRef.current;
        const dot = dotRef.current;
        const spot = spotRef.current;
        if (!ring || !dot || !spot) return;

        // Target (the real pointer) and current (the lagging ring).
        let tx = window.innerWidth / 2;
        let ty = window.innerHeight / 2;
        let cx = tx;
        let cy = ty;
        let raf = 0;
        let running = false;
        let seen = false;

        const root = document.documentElement;

        const tick = () => {
            cx += (tx - cx) * 0.17;
            cy += (ty - cy) * 0.17;
            ring.style.transform = `translate3d(${cx}px, ${cy}px, 0)`;
            dot.style.transform = `translate3d(${tx}px, ${ty}px, 0)`;
            spot.style.setProperty("--mx", `${tx}px`);
            spot.style.setProperty("--my", `${ty}px`);

            if (Math.abs(tx - cx) < 0.35 && Math.abs(ty - cy) < 0.35) {
                running = false;      // caught up: stop burning frames
                return;
            }
            raf = requestAnimationFrame(tick);
        };

        const kick = () => {
            if (running) return;
            running = true;
            raf = requestAnimationFrame(tick);
        };

        const onMove = (e: PointerEvent) => {
            tx = e.clientX;
            ty = e.clientY;
            if (!seen) {
                // Hidden until the pointer exists, so it never flashes mid-screen.
                seen = true;
                cx = tx;
                cy = ty;
                ring.style.opacity = "1";
                dot.style.opacity = "1";
            }
            const el = e.target as Element | null;
            root.classList.toggle(
                "cursor-hot",
                !!(el && typeof el.closest === "function" && el.closest(INTERACTIVE))
            );
            kick();
        };

        const onDown = () => root.classList.add("cursor-down");
        const onUp = () => root.classList.remove("cursor-down");
        const onLeave = () => {
            ring.style.opacity = "0";
            dot.style.opacity = "0";
        };
        const onEnter = () => {
            if (seen) {
                ring.style.opacity = "1";
                dot.style.opacity = "1";
            }
        };
        const onVisibility = () => {
            if (document.hidden) {
                cancelAnimationFrame(raf);
                running = false;
            }
        };

        window.addEventListener("pointermove", onMove, { passive: true });
        window.addEventListener("pointerdown", onDown, { passive: true });
        window.addEventListener("pointerup", onUp, { passive: true });
        document.addEventListener("pointerleave", onLeave);
        document.addEventListener("pointerenter", onEnter);
        document.addEventListener("visibilitychange", onVisibility);

        return () => {
            cancelAnimationFrame(raf);
            window.removeEventListener("pointermove", onMove);
            window.removeEventListener("pointerdown", onDown);
            window.removeEventListener("pointerup", onUp);
            document.removeEventListener("pointerleave", onLeave);
            document.removeEventListener("pointerenter", onEnter);
            document.removeEventListener("visibilitychange", onVisibility);
            root.classList.remove("cursor-hot", "cursor-down");
        };
    }, []);

    return (
        <>
            <div className="ambient-stage" aria-hidden="true">
                <div className="aurora" />
                <div className="blob blob-a" />
                <div className="blob blob-b" />
                <div className="blob blob-c" />
                <div ref={spotRef} className="cursor-spot" />
                <div className="grain" />
            </div>
            <div ref={ringRef} className="cursor-ring" style={{ opacity: 0 }} aria-hidden="true" />
            <div ref={dotRef} className="cursor-dot" style={{ opacity: 0 }} aria-hidden="true" />
        </>
    );
}
