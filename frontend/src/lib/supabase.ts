/**
 * One lazily-created Supabase client, shared by auth and history. Unset env vars
 * leave it inert and the app runs on localStorage. The import is dynamic so
 * ~40 KB never reaches a bundle that will not use it.
 */

export const supabaseConfigured = Boolean(
    process.env.NEXT_PUBLIC_SUPABASE_URL && process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY
);

let clientPromise: Promise<any> | null = null;

export async function getClient(): Promise<any | null> {
    if (!supabaseConfigured || typeof window === "undefined") return null;
    if (!clientPromise) {
        clientPromise = import("@supabase/supabase-js").then((m) =>
            m.createClient(
                process.env.NEXT_PUBLIC_SUPABASE_URL!,
                process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
                { auth: { persistSession: true, detectSessionInUrl: true } }
            )
        );
    }
    return clientPromise;
}

/**
 * The client only when signed in: RLS rejects anonymous inserts, so a guest write
 * would be lost rather than saved locally.
 */
export async function getSignedInClient(): Promise<any | null> {
    const sb = await getClient();
    if (!sb) return null;
    try {
        const { data } = await sb.auth.getSession();
        return data?.session ? sb : null;
    } catch {
        return null;
    }
}
