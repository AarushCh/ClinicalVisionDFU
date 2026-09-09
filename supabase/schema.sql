-- ClinicalVision DFU — analysis history
--
-- Run this once in the Supabase SQL editor, then set these in the FRONTEND
-- build environment (frontend/.env.local, or the deploy workflow):
--
--   NEXT_PUBLIC_SUPABASE_URL=https://<project>.supabase.co
--   NEXT_PUBLIC_SUPABASE_ANON_KEY=<anon key>
--
-- The anon key is public by design. It identifies the project, it does not
-- grant access: every policy below is scoped to auth.uid(), so a signed-in user
-- can only ever read or write their own rows. Never put the SERVICE ROLE key in
-- the frontend — that one does bypass RLS.
--
-- Sign-in is an email magic link (auth.signInWithOtp). For the link to come
-- back into the app, add every origin you serve it from to
-- Authentication -> URL Configuration -> Redirect URLs, e.g.
--   http://localhost:3000/ClinicalVisionDFU
--   https://<user>.github.io/ClinicalVisionDFU
--
-- Nothing here stores identifiable patient data, and the app tells users not to
-- upload any. Only derived numbers and a 160px thumbnail are persisted.

create extension if not exists "pgcrypto";

create table if not exists public.analyses (
    id                    uuid primary key default gen_random_uuid(),
    user_id               uuid not null references auth.users(id) on delete cascade
                              default auth.uid(),
    created_at            timestamptz not null default now(),

    file_name             text,
    thumb                 text,          -- 160px JPEG data URL, ~4 KB

    -- headline outputs
    risk                  text check (risk in ('HIGH','MEDIUM','LOW')),
    risk_probability      real,
    image_probability     real,
    confidence            real,
    predicted_class       text,

    -- clinical stratification
    iwgdf_category        smallint check (iwgdf_category between 0 and 3),
    clinical_logit_shift  real,
    factors_supplied      text[],

    -- explanation readings
    attention_area_pct    real,
    focality              real,

    -- provenance
    architecture          text,
    inference_ms          real
);

create index if not exists analyses_user_created_idx
    on public.analyses (user_id, created_at desc);

alter table public.analyses enable row level security;

-- One policy per verb rather than FOR ALL, so the intent of each is explicit
-- and an accidental broadening of one does not silently widen the others.
drop policy if exists "read own analyses"   on public.analyses;
drop policy if exists "insert own analyses" on public.analyses;
drop policy if exists "update own analyses" on public.analyses;
drop policy if exists "delete own analyses" on public.analyses;

create policy "read own analyses"   on public.analyses
    for select using (auth.uid() = user_id);
create policy "insert own analyses" on public.analyses
    for insert with check (auth.uid() = user_id);
create policy "update own analyses" on public.analyses
    for update using (auth.uid() = user_id) with check (auth.uid() = user_id);
create policy "delete own analyses" on public.analyses
    for delete using (auth.uid() = user_id);

-- Optional: full-resolution scans in Storage rather than data URLs.
-- Private bucket; the same auth.uid() scoping applies, keyed on the first path
-- segment, so objects must be written as "<user-id>/<filename>".
insert into storage.buckets (id, name, public)
values ('scans', 'scans', false)
on conflict (id) do nothing;

drop policy if exists "own scans read"   on storage.objects;
drop policy if exists "own scans write"  on storage.objects;
drop policy if exists "own scans delete" on storage.objects;

create policy "own scans read" on storage.objects
    for select using (
        bucket_id = 'scans' and (storage.foldername(name))[1] = auth.uid()::text
    );
create policy "own scans write" on storage.objects
    for insert with check (
        bucket_id = 'scans' and (storage.foldername(name))[1] = auth.uid()::text
    );
create policy "own scans delete" on storage.objects
    for delete using (
        bucket_id = 'scans' and (storage.foldername(name))[1] = auth.uid()::text
    );
