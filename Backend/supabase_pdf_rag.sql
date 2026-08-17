-- Run this once in the Supabase SQL editor before using PDF Q&A.
create extension if not exists vector;
create extension if not exists pgcrypto;

create table if not exists public.pdf_documents (
  id uuid primary key default gen_random_uuid(),
  user_id text,
  session_id text not null,
  filename text not null,
  file_sha256 text not null,
  page_count integer not null check (page_count between 1 and 30),
  embedding_model text not null default 'nvidia/nemotron-3-embed-1b:free',
  created_at timestamptz not null default now()
);

create table if not exists public.pdf_chunks (
  id uuid primary key default gen_random_uuid(),
  document_id uuid not null references public.pdf_documents(id) on delete cascade,
  user_id text,
  session_id text not null,
  chunk_index integer not null,
  page_start integer not null,
  page_end integer not null,
  chunk_text text not null,
  embedding vector(2048) not null,
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

-- Safe migration for an existing Nexa project. Previous one-shot uploads had
-- no user ownership and are intentionally not exposed to /doc searches.
alter table public.pdf_documents add column if not exists user_id text;
alter table public.pdf_chunks add column if not exists user_id text;
alter table public.pdf_documents drop constraint if exists pdf_documents_page_count_check;
alter table public.pdf_documents
  add constraint pdf_documents_page_count_check check (page_count between 1 and 30);

create index if not exists pdf_chunks_document_idx
  on public.pdf_chunks(document_id, session_id);

create index if not exists pdf_documents_user_idx
  on public.pdf_documents(user_id, created_at desc);

create index if not exists pdf_chunks_user_idx
  on public.pdf_chunks(user_id, created_at desc);

-- Nemotron embeddings are 2048-dimensional. pgvector can store vector(2048),
-- but ivfflat indexes currently cap indexed vector columns at 2000 dimensions.
-- With Nexa's bounded document limits, exact filtered search is fast enough, so no
-- vector index is needed here.

create or replace function public.match_pdf_chunks(
  p_document_id uuid,
  p_session_id text,
  p_query_embedding vector(2048),
  p_match_count integer default 8
)
returns table (
  id uuid,
  document_id uuid,
  chunk_index integer,
  page_start integer,
  page_end integer,
  chunk_text text,
  metadata jsonb,
  similarity double precision
)
language sql
stable
as $$
  select
    c.id,
    c.document_id,
    c.chunk_index,
    c.page_start,
    c.page_end,
    c.chunk_text,
    c.metadata,
    1 - (c.embedding <=> p_query_embedding) as similarity
  from public.pdf_chunks c
  where c.document_id = p_document_id
    and c.session_id = p_session_id
  order by c.embedding <=> p_query_embedding
  limit least(greatest(p_match_count, 1), 20);
$$;

-- Searches only documents explicitly saved with /remember by the signed-in user.
create or replace function public.match_saved_pdf_chunks(
  p_user_id text,
  p_query_embedding vector(2048),
  p_match_count integer default 8
)
returns table (
  id uuid,
  document_id uuid,
  filename text,
  chunk_index integer,
  page_start integer,
  page_end integer,
  chunk_text text,
  metadata jsonb,
  similarity double precision
)
language sql
stable
as $$
  select
    c.id,
    c.document_id,
    d.filename,
    c.chunk_index,
    c.page_start,
    c.page_end,
    c.chunk_text,
    c.metadata,
    1 - (c.embedding <=> p_query_embedding) as similarity
  from public.pdf_chunks c
  join public.pdf_documents d on d.id = c.document_id
  where c.user_id = p_user_id
    and d.user_id = p_user_id
  order by c.embedding <=> p_query_embedding
  limit least(greatest(p_match_count, 1), 20);
$$;

-- Make the new user_id columns and RPC visible to Supabase's REST schema cache
-- immediately after this migration runs.
notify pgrst, 'reload schema';

-- Nexa owner profile: a dedicated application knowledge base. This must not
-- share the user-owned PDF tables above because the resume is private Nexa
-- configuration, not a document belonging to whichever user asks a question.
create table if not exists public.owner_profile_documents (
  profile_key text primary key,
  subject_name text not null,
  source_filename text not null,
  source_sha256 text not null,
  embedding_model text not null,
  chunk_count integer not null default 0 check (chunk_count >= 0),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

alter table public.owner_profile_documents
  add column if not exists chunk_count integer not null default 0;

create table if not exists public.owner_profile_chunks (
  id uuid primary key default gen_random_uuid(),
  profile_key text not null references public.owner_profile_documents(profile_key) on delete cascade,
  source_sha256 text not null,
  chunk_key text not null,
  chunk_index integer not null,
  page_number integer not null check (page_number > 0),
  section text not null,
  title text not null default '',
  chunk_text text not null,
  embedding vector(2048) not null,
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  unique(profile_key, source_sha256, chunk_key)
);

create index if not exists owner_profile_chunks_current_source_idx
  on public.owner_profile_chunks(profile_key, source_sha256, chunk_index);

-- No browser role may read or mutate Nexa's owner profile. The backend uses a
-- Supabase service-role key, which bypasses RLS for this controlled path.
alter table public.owner_profile_documents enable row level security;
alter table public.owner_profile_chunks enable row level security;
revoke all on public.owner_profile_documents from anon, authenticated;
revoke all on public.owner_profile_chunks from anon, authenticated;
grant select, insert, update, delete on public.owner_profile_documents to service_role;
grant select, insert, update, delete on public.owner_profile_chunks to service_role;

create or replace function public.match_owner_profile_chunks(
  p_profile_key text,
  p_query_embedding vector(2048),
  p_match_count integer default 6
)
returns table (
  chunk_key text,
  chunk_index integer,
  page_number integer,
  section text,
  title text,
  chunk_text text,
  similarity double precision
)
language sql
stable
as $$
  select
    c.chunk_key,
    c.chunk_index,
    c.page_number,
    c.section,
    c.title,
    c.chunk_text,
    1 - (c.embedding <=> p_query_embedding) as similarity
  from public.owner_profile_documents d
  join public.owner_profile_chunks c
    on c.profile_key = d.profile_key
   and c.source_sha256 = d.source_sha256
  where d.profile_key = p_profile_key
  order by c.embedding <=> p_query_embedding
  limit least(greatest(p_match_count, 1), 12);
$$;

revoke all on function public.match_owner_profile_chunks(text, vector(2048), integer) from public;
grant execute on function public.match_owner_profile_chunks(text, vector(2048), integer) to service_role;

notify pgrst, 'reload schema';
