# Stock Price Prediction Web App — Project Roadmap

## 1. Folder Structure

```
Stock-Price-Prediction/
├── .env                        # Environment variables (API keys, DB config)
├── .gitignore
├── manage.py                   # Django entry point
├── requirements.txt            # Python dependencies
├── runtime.txt                 # Python version for deployment
├── Procfile                    # Heroku/gunicorn deployment config
│
├── config/                     # Django project settings
│   ├── __init__.py
│   ├── settings.py
│   ├── urls.py                 # Root URL dispatcher
│   ├── wsgi.py
│   └── asgi.py
│
├── app/                        # Main Django app
│   ├── __init__.py
│   ├── admin.py
│   ├── apps.py
│   ├── models.py               # DB models (Stock, Prediction, etc.)
│   ├── views.py                # View logic
│   ├── urls.py                 # App-level URLs
│   ├── serializers.py          # DRF serializers (if using REST API)
│   ├── forms.py                # Django forms (stock symbol input)
│   ├── tasks.py                # Celery tasks for async model training
│   │
│   ├── templates/
│   │   └── app/
│   │       ├── base.html       # Base template
│   │       ├── index.html      # Landing page
│   │       ├── predict.html    # Prediction form + results
│   │       └── compare.html    # Model comparison dashboard
│   │
│   ├── static/
│   │   ├── css/
│   │   │   └── style.css
│   │   ├── js/
│   │   │   └── main.js         # Plotly charts + AJAX calls
│   │   └── images/
│   │
│   ├── ml/                     # Machine Learning module
│   │   ├── __init__.py
│   │   ├── data_fetcher.py     # yfinance data retrieval
│   │   ├── preprocessor.py     # Scaling, sequence creation
│   │   ├── models.py           # ANN and LSTM model definitions
│   │   ├── trainer.py          # Model training pipeline
│   │   ├── predictor.py        # Prediction + inverse transform
│   │   ├── evaluator.py        # RMSE / MAE calculation
│   │   └── utils.py            # Helper functions
│   │
│   └── migrations/
│
├── models_saved/               # Trained model artifacts (.h5 / .keras)
│
├── data/                       # Cached historical data (CSV)
│
└── tests/
    ├── test_models.py
    ├── test_views.py
    ├── test_ml_pipeline.py
    └── test_api.py
```

---

## 2. Development Phases

### Phase 1 — Project Setup & Foundation (Day 1)
- Initialize Django project and app
- Configure `settings.py` (static files, templates, DB, CORS)
- Set up `.env` with `SECRET_KEY`, `DEBUG`, `DATABASE_URL`
- Create base templates and CSS skeleton
- Verify Django runs with `python manage.py runserver`

### Phase 2 — Data Pipeline (Day 2)
- Build `ml/data_fetcher.py` using `yfinance` to download historical OHLCV data
- Build `ml/preprocessor.py`:
  - MinMaxScaler normalization
  - Create sliding-window sequences (e.g., 60 days lookback)
  - Train/test split (80/20 chronological)
- Cache fetched data as CSV in `data/` to avoid redundant API calls
- Handle edge cases: invalid symbol, no data, delisted stocks

### Phase 3 — Model Definition & Training (Day 3–4)
- **ANN model** in `ml/models.py`:
  - Input → Dense(64, relu) → Dropout(0.2) → Dense(32, relu) → Dense(1)
  - Optimizer: Adam, Loss: mse
- **LSTM model** in `ml/models.py`:
  - Input → LSTM(50, return_sequences=True) → Dropout(0.2) → LSTM(50) → Dropout(0.2) → Dense(1)
  - Optimizer: Adam, Loss: mse
- `ml/trainer.py`: Keras `fit()` with early stopping, model checkpoint saving
- `ml/evaluator.py`: Compute RMSE and MAE on test set
- Save trained models to `models_saved/{symbol}_ann.h5` and `models_saved/{symbol}_lstm.h5`

### Phase 4 — Django Views & Templates (Day 5)
- **Home view**: Form to enter stock symbol + date range
- **Predict view**: 
  - Accepts symbol via POST
  - Triggers data fetch → preprocessing → prediction (use cached model if exists, else train)
  - Returns current price + predicted next-day price for both models
- **Compare view**: Table showing RMSE and MAE side-by-side for ANN vs LSTM
- AJAX-based flow: user submits symbol → loading spinner → results appear without full page reload

### Phase 5 — Interactive Charts (Day 6)
- Use **Plotly.js** in templates to render:
  1. **Historical price chart** with 50-day and 200-day moving averages
  2. **Prediction line chart**: actual vs predicted for both ANN and LSTM
  3. **Comparison bar chart**: RMSE and MAE side-by-side
- `static/js/main.js`: Fetch data via Django REST endpoints, feed into Plotly

### Phase 6 — Async Training with Celery (Day 7)
- Install Celery + Redis (or RabbitMQ)
- `tasks.py`: Offload model training as Celery task
- User gets a task ID immediately; frontend polls for completion
- Status endpoint returns `pending / processing / completed / failed`
- Prevents request timeouts during long training runs

### Phase 7 — Testing & Validation (Day 8)
- Unit tests for ML pipeline (data fetch, preprocessing, evaluation)
- Integration tests for views (form submission, prediction output)
- Test edge cases: empty symbol, network failure, very short data
- Validate model accuracy with known stocks

### Phase 8 — Polish & Deploy (Day 9–10)
- Error handling throughout (try/except, user-friendly messages)
- Responsive CSS (mobile-friendly)
- Add `Procfile` and `runtime.txt` for deployment
- Deploy to **Render** / **Heroku** / **AWS EC2**
- Set up CI/CD with GitHub Actions (lint → test → deploy)

---

## 3. Required Libraries

| Package         | Purpose                          |
|-----------------|----------------------------------|
| Django          | Web framework                    |
| djangorestframework | REST API endpoints           |
| django-cors-headers | CORS for frontend-backend    |
| yfinance        | Fetch stock data from Yahoo      |
| tensorflow      | Build and train ANN / LSTM       |
| numpy           | Numerical array operations       |
| pandas          | Data manipulation (DataFrames)   |
| pandas-datareader| Alternative data source (fallback)|
| scikit-learn    | MinMaxScaler, train_test_split   |
| plotly          | Interactive charting (rendered server-side or JSON)|
| celery          | Async task queue                  |
| redis           | Celery broker + cache             |
| gunicorn        | Production WSGI server            |
| python-dotenv   | Environment variable management   |
| whitenoise      | Static file serving in production |
| black / ruff    | Code formatting / linting         |

**requirements.txt:**
```
Django>=5.0,<6.0
djangorestframework>=3.15
django-cors-headers>=4.3
yfinance>=0.2.40
tensorflow>=2.16
numpy>=1.26
pandas>=2.2
scikit-learn>=1.4
plotly>=5.20
celery>=5.3
redis>=5.0
gunicorn>=22.0
python-dotenv>=1.0
whitenoise>=6.6
```

---

## 4. Database Requirements

- **Default**: SQLite (development, zero config)
- **Production**: PostgreSQL (via `psycopg2-binary` or `dj-database-url`)

### Models

**`Stock`**
| Field          | Type          | Notes                        |
|----------------|---------------|------------------------------|
| id             | AutoField     | PK                           |
| symbol         | CharField(10) | Unique, uppercase            |
| name           | CharField(100)| Company name (optional)      |
| last_fetched   | DateTimeField | When data was last retrieved |

**`HistoricalData`**
| Field          | Type          | Notes                        |
|----------------|---------------|------------------------------|
| id             | AutoField     | PK                           |
| stock          | ForeignKey(Stock) | FK to Stock             |
| date           | DateField     | Trading date                 |
| open           | FloatField    |                              |
| high           | FloatField    |                              |
| low            | FloatField    |                              |
| close          | FloatField    | Target variable              |
| volume         | BigIntegerField |                            |
| unique_together |               | (stock, date)                |

**`Prediction`**
| Field          | Type          | Notes                        |
|----------------|---------------|------------------------------|
| id             | AutoField     | PK                           |
| stock          | ForeignKey(Stock) | FK to Stock             |
| model_type     | CharField(10) | 'ANN' or 'LSTM'              |
| prediction_date| DateField     | Date the prediction is for   |
| predicted_price| FloatField    |                              |
| actual_price   | FloatField(null=True) | Filled later if data available |
| rmse           | FloatField    | Test-set RMSE                |
| mae            | FloatField    | Test-set MAE                 |
| created_at     | DateTimeField | Auto now add                 |
| unique_together |               | (stock, model_type, prediction_date) |

---

## 5. API Requirements

### Internal REST Endpoints (DRF)

| Method | Endpoint                     | Description                            |
|--------|------------------------------|----------------------------------------|
| GET    | `/api/stocks/`               | List all tracked stocks                |
| POST   | `/api/fetch-data/`           | Fetch historical data for a symbol     |
| POST   | `/api/predict/`              | Train model(s) and return predictions  |
| GET    | `/api/predictions/<symbol>/` | Get saved predictions for a symbol     |
| GET    | `/api/compare/<symbol>/`     | Get RMSE/MAE for both models           |
| GET    | `/api/task-status/<task_id>/`| Poll Celery task status                |

### External API

- **Yahoo Finance** (via `yfinance`): No API key required.
  - Fetches: `Open`, `High`, `Low`, `Close`, `Volume`
  - Date range: user-configurable (default 5 years)
  - Rate limit: ~2,000 requests/hour (not enforced strictly)

---

## 6. Deployment Strategy

### Platform: Render (recommended) or AWS EC2

| Step | Action |
|------|--------|
| 1    | Push code to GitHub |
| 2    | Create Render Web Service — connect repo |
| 3    | Set build command: `pip install -r requirements.txt && python manage.py migrate && python manage.py collectstatic --noinput` |
| 4    | Set start command: `gunicorn config.wsgi --workers 4 --threads 2` |
| 5    | Add env vars via Render dashboard: `SECRET_KEY`, `DEBUG=False`, `DATABASE_URL`, `REDIS_URL` |
| 6    | Attach Redis instance (Render) for Celery |
| 7    | Create a separate Celery Worker service: `celery -A config worker --loglevel=info` |
| 8    | Set `ALLOWED_HOSTS` in settings |
| 9    | Configure Whitenoise for static files |
| 10   | Set up custom domain + SSL (auto on Render) |

### Alternative: Docker

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD gunicorn config.wsgi --bind 0.0.0.0:$PORT
```

### CI/CD (GitHub Actions)

```yaml
name: CI/CD
on: [push]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.12" }
      - run: pip install -r requirements.txt
      - run: python manage.py test
```

---

## 7. Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| ML lib   | TensorFlow/Keras | Industry standard for sequence models |
| Charts   | Plotly.js (client-side) | Interactive, responsive, no server rendering cost |
| Async    | Celery + Redis | Prevents HTTP timeout during training (10–30 sec) |
| Data source | yfinance | Free, no API key, Python-native |
| Data cache | Local CSV + DB | Avoids re-fetching; CSV for pandas, DB for history |
| Model cache | `.h5` files on disk | Retrain only when user requests or data is stale |
| Scaling  | MinMaxScaler per stock | Each stock has different price range |
| Sequence length | 60 days | Common choice in stock prediction literature |
| Train/test split | 80/20 chronological | Avoids lookahead bias |

---

## 8. Future Enhancements (Post-MVP)

- Technical indicators (RSI, MACD, Bollinger Bands) as features
- Multiple prediction horizons (1d, 7d, 30d)
- Portfolio backtesting simulation
- User authentication + saved watchlists
- GRU / Transformer model comparison
- Sentiment analysis from news headlines
- Real-time dashboard with WebSockets
