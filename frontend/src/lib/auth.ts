"use client";
import { useCallback, useEffect, useState } from "react";
import { getClient, supabaseConfigured } from "./supabase";

/**
 * Guest: a name in localStorage. Personalises the header and scopes history to
 * this browser, and protects nothing. Supabase: a real account, history in
 * Postgres under RLS. Guest is the zero-setup default -- the repo works fully
 * with no Supabase project in existence.
 */
export type Provider = "guest" | "supabase";

export type Session = {
    name: string;
    role: string;
    at: number;
    provider: Provider;
    email?: string;
};

const KEY = "cv-session";
export const ROLES = ["Clinician", "Podiatrist", "Researcher", "Student", "Guest"];

const readGuest = (): Session | null => {
    try {
        const raw = localStorage.getItem(KEY);
        if (!raw) return null;
        const s = JSON.parse(raw);
        return s && typeof s.name === "string" ? { provider: "guest", ...s } : null;
    } catch {
        return null;
    }
};

/** Display name from the account: whatever the user set, else the email local part. */
const fromUser = (user: any): Session => ({
    name:
        user.user_metadata?.name?.trim() ||
        (user.email ? user.email.split("@")[0] : "Account"),
    role: user.user_metadata?.role || "Clinician",
    at: Date.now(),
    provider: "supabase",
    email: user.email,
});

export function useSession() {
    const [session, setSession] = useState<Session | null>(null);
    const [ready, setReady] = useState(false);

    useEffect(() => {
        let alive = true;
        let unsub: (() => void) | undefined;

        (async () => {
            const sb = await getClient();
            if (sb) {
                const { data } = await sb.auth.getSession();
                if (!alive) return;
                if (data?.session?.user) setSession(fromUser(data.session.user));
                else setSession(readGuest());

                // Fires when the magic link lands a session, and on sign-out.
                const sub = sb.auth.onAuthStateChange((_e: string, s: any) => {
                    if (!alive) return;
                    setSession(s?.user ? fromUser(s.user) : readGuest());
                });
                unsub = () => sub?.data?.subscription?.unsubscribe();
            } else {
                setSession(readGuest());
            }
            if (alive) setReady(true);
        })();

        return () => {
            alive = false;
            unsub?.();
        };
    }, []);

    const signInGuest = useCallback((name: string, role: string) => {
        const s: Session = {
            name: name.trim() || "Guest",
            role,
            at: Date.now(),
            provider: "guest",
        };
        try {
            localStorage.setItem(KEY, JSON.stringify(s));
        } catch { }
        setSession(s);
    }, []);

    /** Sends the magic link. The session arrives later via onAuthStateChange. */
    const signInEmail = useCallback(async (email: string, name: string, role: string) => {
        const sb = await getClient();
        if (!sb) throw new Error("Accounts are not configured on this deployment.");
        const { error } = await sb.auth.signInWithOtp({
            email: email.trim(),
            options: {
                emailRedirectTo: window.location.href,
                data: { name: name.trim() || undefined, role },
            },
        });
        if (error) throw new Error(error.message);
    }, []);

    const signOut = useCallback(async () => {
        try {
            localStorage.removeItem(KEY);
        } catch { }
        const sb = await getClient();
        if (sb) await sb.auth.signOut();
        setSession(null);
    }, []);

    return {
        session,
        ready,
        accountsAvailable: supabaseConfigured,
        signInGuest,
        signInEmail,
        signOut,
    };
}
