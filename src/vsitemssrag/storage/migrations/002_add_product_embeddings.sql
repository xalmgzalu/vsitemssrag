create schema if not exists extensions;

create extension if not exists vector with schema extensions;

create table if not exists public.product_embeddings (
    product_id bigint primary key
        references public.products(id) on delete cascade,
    model text not null,
    dimensions smallint not null default 768
        check (dimensions = 768),
    content_hash char(64) not null,
    content text not null,
    embedding extensions.vector(768) not null,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create index if not exists product_embeddings_model_idx
    on public.product_embeddings (model);

create index if not exists product_embeddings_embedding_hnsw_idx
    on public.product_embeddings
    using hnsw (embedding extensions.vector_cosine_ops);

alter table public.product_embeddings enable row level security;

create or replace function public.match_products(
    query_embedding extensions.vector(768),
    query_model text default 'gemini-embedding-2',
    match_count integer default 10,
    only_available boolean default true
)
returns table (
    product_id bigint,
    name text,
    brand text,
    price numeric,
    currency text,
    source_url text,
    characteristics jsonb,
    similarity double precision
)
language sql
stable
security invoker
set search_path = ''
as $$
    select
        products.id,
        products.name,
        products.brand,
        products.price,
        products.currency,
        products.source_url,
        products.characteristics,
        1 - (
            embeddings.embedding
            operator(extensions.<=>)
            query_embedding
        ) as similarity
    from public.product_embeddings as embeddings
    join public.products as products on products.id = embeddings.product_id
    where embeddings.model = query_model
      and (not only_available or products.available)
    order by embeddings.embedding operator(extensions.<=>) query_embedding
    limit greatest(least(match_count, 100), 0)
$$;

revoke all on function public.match_products(
    extensions.vector,
    text,
    integer,
    boolean
) from public;
