"use client";
import { useState } from "react";
import ThemeToggle from "./ThemeToggle";
import Ambient from "./Ambient";
import { ROLES } from "@/lib/auth";

/**
 * Account: Supabase magic link, history in Postgres under RLS. Guest: a name in
 * localStorage that protects nothing and says so. Neither authorises /predict --
 * the backend is open by design for a research prototype.
 */
export default function AuthGate({
    accountsAvailable, onGuest, onEmail,
}: {
    accountsAvailable: boolean;
    onGuest: (name: string, role: string) => void;
    onEmail: (email: string, name: string, role: string) => Promise<void>;
}) {
    const [mode, setMode] = useState<"account" | "guest">(accountsAvailable ? "account" : "guest");
    const [name, setName] = useState("");
    const [role, setRole] = useState(ROLES[0]);
    const [email, setEmail] = useState("");
    const [busy, setBusy] = useState(false);
    const [sent, setSent] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const submitEmail = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!email.trim()) return;
        setBusy(true);
        setError(null);
        try {
            await onEmail(email, name, role);
            setSent(true);
        } catch (err: any) {
            setError(err?.message || "Could not send the sign-in link.");
        } finally {
            setBusy(false);
        }
    };

    return (
        <main className="min-h-screen bg-bg text-fg grid lg:grid-cols-2 relative overflow-hidden">
            <Ambient />

            <div className="absolute top-5 right-5 z-20">
                <ThemeToggle />
            </div>

            {/* Left: what this is. Empty space on a sign-in screen is wasted space. */}
            <section className="hidden lg:flex flex-col justify-center px-14 xl:px-20 relative z-10 bento-grid-bg">
                <div className="flex items-center gap-3 mb-8">
                    <div className="p-2.5 bg-brand/20 rounded-xl border border-brand/30 text-brand">
                        <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.8} d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z" />
                        </svg>
                    </div>
                    <div>
                        <h1 className="text-lg font-black tracking-tight">ClinicalVision DFU</h1>
                        <p className="text-[11px] text-subtle">Explainable AI triage</p>
                    </div>
                </div>

                <h2 className="text-4xl xl:text-5xl font-black tracking-tight leading-[1.05] mb-5">
                    Every number on the<br />report is <span className="text-gradient">traceable</span>.
                </h2>
                <p className="text-sm text-muted leading-relaxed max-w-md mb-10">
                    A calibrated CNN classifies the tissue, Grad-CAM++ shows where it looked,
                    and the clinical history is fused in log-odds space with exact Shapley
                    attribution — so the risk score can be taken apart factor by factor.
                </p>

                <dl className="grid grid-cols-3 gap-4 max-w-md">
                    {[
                        ["98.1%", "test accuracy"],
                        ["0.997", "AUROC"],
                        ["~10 ms", "CPU inference"],
                    ].map(([v, k]) => (
                        <div key={k} className="panel-inset rounded-xl p-3.5">
                            <dd className="text-xl font-black tabular-nums leading-none">{v}</dd>
                            <dt className="text-[10px] text-subtle uppercase tracking-[0.12em] mt-1.5">{k}</dt>
                        </div>
                    ))}
                </dl>

                <p className="text-[10px] text-subtle mt-8 max-w-md leading-relaxed">
                    Research prototype. Not a medical device, not FDA/CE cleared, and not
                    validated against clinical outcomes.
                </p>
            </section>

            {/* Right: the actual gate. */}
            <section className="flex items-center justify-center p-6 relative z-10">
                <div className="panel rounded-2xl p-7 w-full max-w-sm animate-fadeUp">
                    <div className="lg:hidden flex items-center gap-2.5 mb-6">
                        <div className="p-2 bg-brand/20 rounded-lg border border-brand/30 text-brand">
                            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.8} d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z" />
                            </svg>
                        </div>
                        <div>
                            <h1 className="font-black tracking-tight leading-none">ClinicalVision DFU</h1>
                            <p className="text-[10px] text-subtle">Explainable AI triage</p>
                        </div>
                    </div>

                    {accountsAvailable && (
                        <div className="flex gap-1 panel-inset p-1 rounded-xl mb-6">
                            {(["account", "guest"] as const).map((m) => (
                                <button key={m} type="button" onClick={() => { setMode(m); setError(null); }}
                                    className={`flex-1 py-2 rounded-lg text-[10px] font-black uppercase tracking-[0.12em] transition-colors ${mode === m ? "bg-brand/20 text-brand" : "text-subtle hover:text-fg"}`}>
                                    {m === "account" ? "Account" : "Guest"}
                                </button>
                            ))}
                        </div>
                    )}

                    {sent ? (
                        <div className="text-center py-6">
                            <div className="w-12 h-12 rounded-xl bg-good/15 border border-good/30 text-good grid place-items-center mx-auto mb-4">
                                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                                </svg>
                            </div>
                            <p className="font-bold text-sm mb-1.5">Check your inbox</p>
                            <p className="text-xs text-muted leading-relaxed">
                                A sign-in link is on its way to <strong className="text-fg">{email}</strong>.
                                Open it on this device and you will land back here, signed in.
                            </p>
                            <button type="button" onClick={() => setSent(false)}
                                className="text-[11px] text-subtle hover:text-fg mt-5 transition-colors">
                                Use a different address
                            </button>
                        </div>
                    ) : mode === "account" ? (
                        <form className="space-y-4" onSubmit={submitEmail}>
                            <div>
                                <h2 className="font-black tracking-tight mb-1">Sign in</h2>
                                <p className="text-xs text-muted leading-relaxed">
                                    No password. We email a one-time link; your analysis history
                                    then syncs across devices and is readable only by you.
                                </p>
                            </div>

                            <div>
                                <label htmlFor="cv-email" className="block text-[10px] font-bold text-subtle uppercase tracking-[0.1em] mb-1.5">
                                    Email
                                </label>
                                <input id="cv-email" type="email" required autoComplete="email"
                                    className="field py-2.5 text-sm" value={email}
                                    onChange={(e) => setEmail(e.target.value)}
                                    placeholder="you@hospital.org" />
                            </div>

                            <div className="grid grid-cols-2 gap-2.5">
                                <div>
                                    <label htmlFor="cv-name" className="block text-[10px] font-bold text-subtle uppercase tracking-[0.1em] mb-1.5">
                                        Name
                                    </label>
                                    <input id="cv-name" className="field py-2.5 text-sm" value={name}
                                        onChange={(e) => setName(e.target.value)}
                                        placeholder="Optional" autoComplete="name" maxLength={60} />
                                </div>
                                <div>
                                    <label htmlFor="cv-role" className="block text-[10px] font-bold text-subtle uppercase tracking-[0.1em] mb-1.5">
                                        Role
                                    </label>
                                    <select id="cv-role" className="field py-2.5 text-sm appearance-none cursor-pointer"
                                        value={role} onChange={(e) => setRole(e.target.value)}>
                                        {ROLES.map((r) => <option key={r} value={r}>{r}</option>)}
                                    </select>
                                </div>
                            </div>

                            {error && <p role="alert" className="text-[11px] text-danger leading-snug">{error}</p>}

                            <button type="submit" disabled={busy}
                                className="w-full rounded-xl px-5 py-3 bg-brand text-brandFg font-black text-xs tracking-[0.15em] uppercase hover:opacity-90 transition-opacity disabled:opacity-50">
                                {busy ? "Sending…" : "Email me a link"}
                            </button>

                            <button type="button" onClick={() => setMode("guest")}
                                className="w-full text-[11px] text-subtle hover:text-fg transition-colors">
                                or continue without an account
                            </button>
                        </form>
                    ) : (
                        <form className="space-y-4" onSubmit={(e) => { e.preventDefault(); onGuest(name, role); }}>
                            <div>
                                <h2 className="font-black tracking-tight mb-1">Continue as guest</h2>
                                <p className="text-xs text-muted leading-relaxed">
                                    Nothing to sign up for. Your history stays in this browser.
                                </p>
                            </div>

                            {/* A login implying security it lacks is worse than none. */}
                            <p className="flex items-start gap-2 px-3 py-2.5 rounded-lg bg-warn/10 border border-warn/25 text-[11px] text-warn leading-snug">
                                <svg className="w-3.5 h-3.5 shrink-0 mt-px" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01M5.07 19h13.86a2 2 0 001.71-3.03l-6.93-12a2 2 0 00-3.42 0l-6.93 12A2 2 0 005.07 19z" />
                                </svg>
                                <span>
                                    <strong>Not authentication.</strong> No password is checked and
                                    nothing is protected. Never enter real patient data.
                                </span>
                            </p>

                            <div className="grid grid-cols-2 gap-2.5">
                                <div>
                                    <label htmlFor="cv-gname" className="block text-[10px] font-bold text-subtle uppercase tracking-[0.1em] mb-1.5">
                                        Name
                                    </label>
                                    <input id="cv-gname" className="field py-2.5 text-sm" value={name}
                                        onChange={(e) => setName(e.target.value)}
                                        placeholder="e.g. A. Udbhav" autoComplete="name" maxLength={60} />
                                </div>
                                <div>
                                    <label htmlFor="cv-grole" className="block text-[10px] font-bold text-subtle uppercase tracking-[0.1em] mb-1.5">
                                        Role
                                    </label>
                                    <select id="cv-grole" className="field py-2.5 text-sm appearance-none cursor-pointer"
                                        value={role} onChange={(e) => setRole(e.target.value)}>
                                        {ROLES.map((r) => <option key={r} value={r}>{r}</option>)}
                                    </select>
                                </div>
                            </div>

                            <button type="submit"
                                className="w-full rounded-xl px-5 py-3 bg-brand text-brandFg font-black text-xs tracking-[0.15em] uppercase hover:opacity-90 transition-opacity">
                                Open dashboard
                            </button>

                            {!accountsAvailable && (
                                <p className="text-[10px] text-subtle leading-relaxed pt-1">
                                    Accounts appear here once the deployment has
                                    <code className="text-muted"> NEXT_PUBLIC_SUPABASE_URL </code>
                                    and an anon key — see <code className="text-muted">supabase/schema.sql</code>.
                                </p>
                            )}
                        </form>
                    )}
                </div>
            </section>
        </main>
    );
}
