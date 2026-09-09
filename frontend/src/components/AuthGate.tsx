"use client";
import { useEffect, useState } from "react";
import ThemeToggle from "./ThemeToggle";

export type Session = { name: string; role: string; at: number };

const KEY = "cv-session";
const ROLES = ["Clinician", "Podiatrist", "Researcher", "Student", "Guest"];

/**
 * DEMO ACCESS GATE — deliberately not real authentication.
 *
 * This frontend is a static export (next.config.mjs sets output:"export") served
 * from GitHub Pages. There is no session store, no password verification and no
 * server-side authorisation anywhere in this path: the "session" is a plain
 * localStorage object that any visitor can write from the console.
 *
 * It is here to demonstrate the sign-in *flow* and to personalise the report
 * header. It protects nothing, and it is labelled as such on screen so no one
 * mistakes it for access control. Real auth would need the backend to own
 * sessions (e.g. OAuth + signed cookies) and to authorise /predict per request —
 * see §19 of ClinicalVisionDFU_Summary.md.
 */
export function useSession() {
    const [session, setSession] = useState<Session | null>(null);
    const [ready, setReady] = useState(false);

    useEffect(() => {
        try {
            const raw = localStorage.getItem(KEY);
            if (raw) setSession(JSON.parse(raw));
        } catch {
            // corrupt or unavailable storage: fall through to signed-out
        }
        setReady(true);
    }, []);

    const signIn = (name: string, role: string) => {
        const s: Session = { name: name.trim() || "Guest", role, at: Date.now() };
        try {
            localStorage.setItem(KEY, JSON.stringify(s));
        } catch { }
        setSession(s);
    };

    const signOut = () => {
        try {
            localStorage.removeItem(KEY);
        } catch { }
        setSession(null);
    };

    return { session, ready, signIn, signOut };
}

export default function AuthGate({
    onSignIn,
}: {
    onSignIn: (name: string, role: string) => void;
}) {
    const [name, setName] = useState("");
    const [role, setRole] = useState(ROLES[0]);

    return (
        <main className="min-h-screen bg-bg text-fg flex items-center justify-center p-6 relative overflow-hidden">
            <div className="ambient fixed inset-0 pointer-events-none z-0 overflow-hidden">
                <div className="absolute top-[-20%] left-[-10%] w-[50%] h-[50%] bg-brand/15 blur-[150px] rounded-full" />
                <div className="absolute bottom-[-20%] right-[-10%] w-[40%] h-[40%] bg-blue-500/15 blur-[150px] rounded-full" />
            </div>

            <div className="absolute top-6 right-6 z-20">
                <ThemeToggle />
            </div>

            <div className="panel rounded-[2rem] p-8 sm:p-10 w-full max-w-md relative z-10 animate-fadeUp">
                <div className="flex items-center gap-3 mb-2">
                    <div className="p-2.5 bg-brand/20 rounded-xl border border-brand/30 text-brand">
                        <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.8} d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z" />
                        </svg>
                    </div>
                    <div>
                        <h1 className="text-xl font-black tracking-tight">ClinicalVision DFU</h1>
                        <p className="text-[11px] text-subtle font-medium">Explainable AI triage</p>
                    </div>
                </div>

                <p className="text-sm text-muted leading-relaxed mt-5 mb-6">
                    Enter a name and role to open the dashboard. This personalises the
                    report header and nothing else.
                </p>

                {/* Stating this plainly matters more than the feature does. A
                    medical-looking login that implies security it does not have is
                    worse than no login at all. */}
                <div className="flex items-start gap-2.5 px-4 py-3 rounded-xl bg-warn/10 border border-warn/30 mb-6">
                    <svg className="w-4 h-4 text-warn shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01M5.07 19h13.86a2 2 0 001.71-3.03l-6.93-12a2 2 0 00-3.42 0l-6.93 12A2 2 0 005.07 19z" />
                    </svg>
                    <p className="text-[11px] text-warn leading-relaxed">
                        <strong>Demo access, not authentication.</strong> No password is
                        checked and nothing is protected — this is a static site with no
                        session server. Do not enter real patient data anywhere in this
                        application.
                    </p>
                </div>

                <form
                    className="space-y-5"
                    onSubmit={(e) => {
                        e.preventDefault();
                        onSignIn(name, role);
                    }}
                >
                    <div className="space-y-2">
                        <label htmlFor="cv-name" className="text-[10px] font-black text-brand uppercase tracking-[0.15em]">
                            Your name
                        </label>
                        <input
                            id="cv-name"
                            className="field"
                            value={name}
                            onChange={(e) => setName(e.target.value)}
                            placeholder="e.g. A. Udbhav"
                            autoComplete="name"
                            maxLength={60}
                        />
                    </div>

                    <div className="space-y-2">
                        <label htmlFor="cv-role" className="text-[10px] font-black text-brand uppercase tracking-[0.15em]">
                            Role
                        </label>
                        <select
                            id="cv-role"
                            className="field appearance-none cursor-pointer"
                            value={role}
                            onChange={(e) => setRole(e.target.value)}
                        >
                            {ROLES.map((r) => (
                                <option key={r} value={r}>{r}</option>
                            ))}
                        </select>
                    </div>

                    <button
                        type="submit"
                        className="w-full rounded-xl px-6 py-4 bg-brand text-brandFg font-bold text-sm tracking-wider uppercase hover:opacity-90 transition-opacity"
                    >
                        Enter dashboard
                    </button>
                </form>

                <button
                    type="button"
                    onClick={() => onSignIn("Guest", "Guest")}
                    className="w-full mt-3 text-xs text-subtle hover:text-fg transition-colors py-2"
                >
                    Continue as guest
                </button>
            </div>
        </main>
    );
}
