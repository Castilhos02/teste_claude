# Music Recommendation API

API de recomendação musical baseada em análise de grafos, construída com FastAPI, NetworkX e scikit-learn. Criado somente para testar a funcionalidade do Claude, utilizando poucos comandos para criar algo complexo com uma ferramente de IA nova no mercado. Em processo de análise para entender os recursos a serem implementados para a melhoria dos prompts. 

## Stack

| Camada | Tecnologia |
|---|---|
| Framework | FastAPI + lifespan |
| Grafo | NetworkX (bipartido multipartido) |
| ML | scikit-learn (SVD, cosine similarity) |
| Cache | Redis (fallback in-memory) |
| Auth | JWT (python-jose + bcrypt) |
| Validação | Pydantic v2 + SecretStr |
| Logging | structlog (JSON em produção) |
| Testes | pytest-asyncio + httpx |
| Container | Docker multi-stage |
| CI/CD | GitHub Actions |

## Algoritmos de recomendação

### 1. Personalized PageRank
Executa PageRank no grafo de interações com vetor de personalização baseado no histórico do usuário. Propaga relevância para tracks vizinhas na rede.

### 2. Content-Based (cosine similarity)
Calcula o perfil sonoro do usuário como média dos vetores de audio features das tracks curtidas, e encontra as tracks mais similares usando similaridade cosine.

### 3. Collaborative Filtering (SVD)
Fatoriza a matriz usuário-item usando SVD truncado (scikit-learn). Reconstrói preferências latentes e pontua tracks não ouvidas.

### 4. Embedding Similarity
Similaridade direta entre embeddings de usuário e track. Compatível com embeddings externos (Word2Vec, BERT, etc.).

### 5. Hybrid (padrão)
Combina todos os algoritmos com pesos ponderados e normalização por ranking.

## Estrutura do projeto

```
music-api/
├── app/
│   ├── main.py                    # App factory + lifespan
│   ├── core/
│   │   ├── config.py              # BaseSettings + SecretStr
│   │   ├── security.py            # JWT + bcrypt
│   │   ├── exceptions.py          # Hierarquia de exceções tipadas
│   │   └── logging.py             # structlog configurado
│   ├── domain/
│   │   └── models.py              # Pydantic v2 — User, Track, Interaction, ...
│   ├── graph/
│   │   ├── engine.py              # MusicGraph (NetworkX)
│   │   └── algorithms.py          # PageRank, SVD, cosine, hybrid
│   ├── repositories/
│   │   └── base.py                # Interfaces + InMemory implementations
│   ├── services/
│   │   ├── recommendation.py      # Orquestração dos algoritmos
│   │   └── cache.py               # Redis + in-memory fallback
│   └── api/
│       ├── dependencies.py        # Injeção de dependência (Annotated)
│       ├── middleware/
│       │   └── error_handler.py   # Logging + exception handlers
│       └── routes/
│           ├── auth.py            # POST /register, /login
│           ├── recommendations.py # POST /recommendations, /interactions
│           ├── tracks.py          # CRUD /tracks
│           └── health.py          # GET /health, /ready, /live
├── tests/
│   ├── conftest.py
│   ├── unit/
│   │   ├── test_graph_algorithms.py
│   │   ├── test_security.py
│   │   └── test_domain_models.py
│   └── integration/
│       └── test_api.py
├── .github/workflows/ci.yml
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── .env.example
```

## Início rápido

```bash
# 1. Clonar e configurar
cp .env.example .env
# Edite .env com seu JWT_SECRET_KEY forte

# 2. Docker (recomendado)
docker compose up -d

# 3. Ou local
pip install -e ".[dev]"
uvicorn app.main:app --reload

# 4. Docs interativos
open http://localhost:8000/docs
```

## Testes

```bash
# Todos os testes com cobertura
pytest tests/ -v --cov=app

# Apenas unitários
pytest tests/unit/ -v

# Apenas integração
pytest tests/integration/ -v
```

## Docker

```bash
# Build
docker build --target runtime -t music-api .

# Stack completa com Redis
docker compose up -d

# Rodar testes em container
docker compose --profile test run tests

# Redis Insight (UI para Redis)
docker compose --profile observability up -d redis-insight
# Acesse: http://localhost:5540
```

## Endpoints principais

| Método | Rota | Descrição |
|---|---|---|
| POST | /api/v1/auth/register | Cadastro |
| POST | /api/v1/auth/login | Login → tokens JWT |
| POST | /api/v1/recommendations | Gerar recomendações |
| POST | /api/v1/recommendations/interactions | Registrar interação |
| POST | /api/v1/tracks | Criar track |
| GET | /api/v1/tracks | Listar tracks |
| GET | /api/v1/tracks/{id} | Buscar track |
| GET | /health | Health check completo |

## Decisões de arquitetura

**Repositórios com interface abstrata** — `InMemoryUserRepository` implementa `AbstractUserRepository`. Para trocar por PostgreSQL, implemente a interface e injete via `app.state`. A camada de serviço não muda.

**Cache com fallback transparente** — Se o Redis não estiver disponível, a API usa cache in-memory automaticamente. Sem downtime, sem exceção.

**SecretStr em toda credencial** — `JWT_SECRET_KEY`, `REDIS_URL` e senhas nunca aparecem em logs, stack traces ou serialização JSON acidental.

**lifespan em vez de on_event** — O padrão moderno do FastAPI garante startup/shutdown atômico com `async with`, sem vazamento de recursos.

**Exceções tipadas com status_code** — Toda exceção de domínio carrega seu HTTP status code. O handler centralizado converte para JSON padronizado sem `try/except` nos endpoints.

## Próximos passos para produção

- [ ] Substituir `InMemoryRepository` por SQLAlchemy + PostgreSQL
- [ ] Adicionar Neo4j como backend do grafo
- [ ] Implementar refresh token rotation com Redis blacklist
- [ ] Adicionar OpenTelemetry + Prometheus metrics
- [ ] Rate limiting por usuário (não só por IP)
- [ ] Background task para recalcular embeddings periodicamente
