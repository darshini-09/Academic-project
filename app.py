from __future__ import annotations

import json
import os
from pathlib import Path
from functools import wraps

import joblib
import pandas as pd
from flask import Flask, render_template, request, session, redirect, url_for
from werkzeug.security import check_password_hash, generate_password_hash

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "change-this-secret-key")

MODEL_PATH = Path("model/layoff_model.pkl")
META_PATH = Path("model/model_meta.pkl")
DATA_PATH = Path("Data/processed/Training_Data.csv")
USERS_PATH = Path("Data/users.json")
TRAIN_PATH = Path("Data/processed/Training_Data.csv")
TEST_PATH = Path("Data/processed/Testing_Data.csv")

model = None
meta = None
choices = {}
decision_threshold = 0.5
ALLOWED_ROLES = {"hr_analyst", "hr_manager", "system_admin"}
DEFAULT_ROLE = "hr_analyst"


def _to_float(form, key: str, default: float = 0.0) -> float:
    value = form.get(key, "").strip()
    if value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _read_uploaded_table(file_storage) -> pd.DataFrame:
    filename = (file_storage.filename or "").lower()
    if filename.endswith(".csv"):
        return pd.read_csv(file_storage)
    if filename.endswith(".xlsx") or filename.endswith(".xls"):
        return pd.read_excel(file_storage)
    raise ValueError("Unsupported file type. Upload .csv or .xlsx")


def _dataset_options() -> dict[str, Path]:
    return {
        "training": TRAIN_PATH,
        "testing": TEST_PATH,
    }


def _to_typed_value(value: str, series: pd.Series):
    if pd.api.types.is_numeric_dtype(series.dtype):
        if value == "":
            return pd.NA
        num = pd.to_numeric(value, errors="coerce")
        return num
    return value


def _load_users() -> dict:
    if not USERS_PATH.exists():
        return {}
    try:
        return json.loads(USERS_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _save_users(users: dict) -> None:
    USERS_PATH.parent.mkdir(parents=True, exist_ok=True)
    USERS_PATH.write_text(json.dumps(users, indent=2), encoding="utf-8")


def _has_any_user() -> bool:
    return len(_load_users()) > 0


def login_required(view_func):
    @wraps(view_func)
    def wrapper(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect(url_for("login"))
        return view_func(*args, **kwargs)

    return wrapper


def load_assets() -> None:
    global model, meta, choices, decision_threshold

    if not MODEL_PATH.exists() or not META_PATH.exists():
        raise FileNotFoundError(
            "Model files missing. Run: python scripts/train_business_model.py"
        )

    model = joblib.load(MODEL_PATH)
    meta = joblib.load(META_PATH)
    decision_threshold = float(meta.get("decision_threshold", 0.5))

    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
        for col in meta["cat_features"]:
            if col in df.columns:
                vals = sorted(df[col].dropna().astype(str).unique().tolist())
                choices[col] = vals


@app.route("/login", methods=["GET", "POST"])
def login():
    embedded = request.args.get("embed") == "1" or request.form.get("embed") == "1"
    if not _has_any_user():
        return redirect(url_for("register", embed="1" if embedded else None))

    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        users = _load_users()
        hashed = users.get(username)
        if hashed and check_password_hash(hashed, password):
            session["logged_in"] = True
            session["username"] = username
            if embedded:
                return (
                    "<script>window.top.location = '/dashboard';</script>",
                    200,
                    {"Content-Type": "text/html"},
                )
            return redirect(url_for("dashboard"))
        error = "Invalid username or password."
    return render_template("login.html", error=error, embedded=embedded)


@app.route("/register", methods=["GET", "POST"])
def register():
    embedded = request.args.get("embed") == "1" or request.form.get("embed") == "1"
    error = None
    success = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        confirm = request.form.get("confirm_password", "").strip()

        if len(username) < 3:
            error = "Username must be at least 3 characters."
        elif len(password) < 6:
            error = "Password must be at least 6 characters."
        elif password != confirm:
            error = "Passwords do not match."
        else:
            users = _load_users()
            if username in users:
                error = "Username already exists. Please choose another."
            else:
                users[username] = generate_password_hash(password)
                _save_users(users)
                success = "Registration successful. Please login."
                return redirect(url_for("login", embed="1" if embedded else None))

    return render_template("register.html", error=error, success=success, embedded=embedded)


@app.route("/logout", methods=["GET"])
def logout():
    session.clear()
    return redirect(url_for("home"))


@app.route("/", methods=["GET"])
def home():
    return render_template(
        "landing.html",
        logged_in=bool(session.get("logged_in")),
        username=session.get("username", ""),
    )


@app.route("/dashboard", methods=["GET"])
@login_required
def dashboard():
    active_page = request.args.get("tab", "single")
    if active_page not in {"single", "batch", "records", "graphs"}:
        active_page = "single"
    return render_template(
        "dashboard.html",
        username=session.get("username", "User"),
        active_page=active_page,
    )


@app.route("/single", methods=["GET", "POST"])
@login_required
def single_evaluation():
    embedded = request.args.get("embed") == "1"
    prediction = None
    probability = None
    error = None

    defaults = {
        "Age": "35",
        "DistanceFromHome": "10",
        "monthly_income": "5000",
        "NumCompaniesWorked": "2",
        "PercentSalaryHike": "13",
        "PerformanceRating": "3",
        "YearsInCurrentRole": "4",
        "YearsSinceLastPromotion": "1",
        "JobSatisfaction": "3",
        "BusinessTravel": "Travel_Rarely",
        "department": "Research & Development",
        "EducationField": "Life Sciences",
        "Gender": "Male",
        "MaritalStatus": "Single",
        "job_role": "Research Scientist",
        "overtime": "No",
        "industry_proxy": "Technology and Services",
        "national_workers_affected": "1500",
    }

    form_values = defaults.copy()

    if request.method == "POST":
        form_values.update({k: request.form.get(k, defaults.get(k, "")) for k in defaults})
        try:
            row = {
                "Age": _to_float(request.form, "Age", 35),
                "DistanceFromHome": _to_float(request.form, "DistanceFromHome", 10),
                "monthly_income": _to_float(request.form, "monthly_income", 5000),
                "NumCompaniesWorked": _to_float(request.form, "NumCompaniesWorked", 2),
                "PercentSalaryHike": _to_float(request.form, "PercentSalaryHike", 13),
                "PerformanceRating": _to_float(request.form, "PerformanceRating", 3),
                "YearsInCurrentRole": _to_float(request.form, "YearsInCurrentRole", 4),
                "YearsSinceLastPromotion": _to_float(request.form, "YearsSinceLastPromotion", 1),
                "JobSatisfaction": _to_float(request.form, "JobSatisfaction", 3),
                "BusinessTravel": request.form.get("BusinessTravel", defaults["BusinessTravel"]),
                "department": request.form.get("department", defaults["department"]),
                "EducationField": request.form.get("EducationField", defaults["EducationField"]),
                "Gender": request.form.get("Gender", defaults["Gender"]),
                "MaritalStatus": request.form.get("MaritalStatus", defaults["MaritalStatus"]),
                "job_role": request.form.get("job_role", defaults["job_role"]),
                "overtime": request.form.get("overtime", defaults["overtime"]),
                "industry_proxy": request.form.get("industry_proxy", defaults["industry_proxy"]),
                "national_workers_affected": _to_float(request.form, "national_workers_affected", 1500),
            }

            input_df = pd.DataFrame([row])
            prob = float(model.predict_proba(input_df)[0][1])
            pred = 1 if prob >= decision_threshold else 0

            probability = round(prob * 100, 2)
            prediction = "Layoff Risk: High" if pred == 1 else "Layoff Risk: Low"
        except Exception as exc:
            error = str(exc)

    return render_template(
        "single.html",
        prediction=prediction,
        probability=probability,
        error=error,
        form_values=form_values,
        choices=choices,
        required_features=meta["features"],
        username=session.get("username", "User"),
        class_balance=meta.get("class_balance", {}),
        active_page="single",
        embedded=embedded,
    )


@app.route("/batch", methods=["GET", "POST"])
@login_required
def batch_evaluation():
    embedded = request.args.get("embed") == "1" or request.form.get("embed") == "1"
    defaults = {
        "Age": "35",
        "DistanceFromHome": "10",
        "monthly_income": "5000",
        "NumCompaniesWorked": "2",
        "PercentSalaryHike": "13",
        "PerformanceRating": "3",
        "YearsInCurrentRole": "4",
        "YearsSinceLastPromotion": "1",
        "JobSatisfaction": "3",
        "BusinessTravel": "Travel_Rarely",
        "department": "Research & Development",
        "EducationField": "Life Sciences",
        "Gender": "Male",
        "MaritalStatus": "Single",
        "job_role": "Research Scientist",
        "overtime": "No",
        "industry_proxy": "Technology and Services",
        "national_workers_affected": "1500",
    }
    batch_error = None
    table_rows = []
    table_columns = []
    summary = None

    if request.method == "POST":
        try:
            if "dataset_file" not in request.files:
                raise ValueError("No file selected.")
            uploaded = request.files["dataset_file"]
            if not uploaded.filename:
                raise ValueError("No file selected.")

            df = _read_uploaded_table(uploaded)
            required = meta["features"]
            missing = [c for c in required if c not in df.columns]
            if missing:
                raise ValueError("Missing required columns: " + ", ".join(missing))

            X = df[required].copy()
            prob = model.predict_proba(X)[:, 1]
            pred = (prob >= decision_threshold).astype(int)

            out = df.copy()
            out["layoff_probability"] = (prob * 100).round(2)
            out["layoff_prediction"] = ["High" if p == 1 else "Low" for p in pred]

            total_rows = len(out)
            high_count = int((out["layoff_prediction"] == "High").sum())
            low_count = int((out["layoff_prediction"] == "Low").sum())
            avg_risk = float(out["layoff_probability"].mean()) if total_rows > 0 else 0.0
            summary = {
                "total_rows": total_rows,
                "high_count": high_count,
                "low_count": low_count,
                "avg_risk": round(avg_risk, 2),
            }

            preview = out.head(200).copy()
            table_columns = preview.columns.tolist()
            table_rows = preview.to_dict(orient="records")
        except Exception as exc:
            batch_error = str(exc)

    return render_template(
        "batch.html",
        batch_error=batch_error,
        form_values=defaults,
        choices=choices,
        required_features=meta["features"],
        username=session.get("username", "User"),
        class_balance=meta.get("class_balance", {}),
        table_rows=table_rows,
        table_columns=table_columns,
        summary=summary,
        active_page="batch",
        embedded=embedded,
    )


@app.route("/records", methods=["GET", "POST"])
@login_required
def records():
    embedded = request.args.get("embed") == "1" or request.form.get("embed") == "1"
    message = None
    error = None
    record = None
    total_rows = 0
    dataset_key = request.values.get("dataset", "training")
    row_index_query = request.values.get("row_index", "").strip()
    action = request.form.get("action", "").strip()

    datasets = _dataset_options()
    if dataset_key not in datasets:
        dataset_key = "training"
    dataset_path = datasets[dataset_key]

    if not dataset_path.exists():
        error = f"Dataset file not found: {dataset_path}"
        return render_template(
            "records.html",
            username=session.get("username", "User"),
            active_page="records",
            datasets=list(datasets.keys()),
            dataset_key=dataset_key,
            record=record,
            row_index_query=row_index_query,
            total_rows=total_rows,
            message=message,
            error=error,
            embedded=embedded,
        )

    df = pd.read_csv(dataset_path)
    editable_columns = [c for c in df.columns if c not in {"layoff_probability", "layoff_prediction"}]
    total_rows = len(df)

    if total_rows > 0 and row_index_query == "":
        row_index_query = "0"

    if row_index_query != "":
        try:
            idx = int(row_index_query)
        except ValueError:
            idx = -1
        if idx < 0 or idx >= len(df):
            error = "Invalid row index selected."
        else:
            row = df.loc[idx, editable_columns].to_dict()
            record = {"row_index": idx, "values": {k: "" if pd.isna(v) else str(v) for k, v in row.items()}}

    if action == "save":
        row_index_raw = request.form.get("row_index", "").strip()
        if row_index_raw == "":
            error = "Record index missing. Load a record first."
        else:
            try:
                row_index = int(row_index_raw)
            except ValueError:
                row_index = -1
            if row_index < 0 or row_index >= len(df):
                error = "Invalid record index."
            else:
                for col in editable_columns:
                    incoming = request.form.get(col, "")
                    df.at[row_index, col] = _to_typed_value(incoming, df[col])
                df.to_csv(dataset_path, index=False)
                message = f"Record updated successfully in {dataset_path.name}."
                updated = df.loc[row_index, editable_columns].to_dict()
                record = {
                    "row_index": row_index,
                    "values": {k: "" if pd.isna(v) else str(v) for k, v in updated.items()},
                }

    if action == "delete":
        row_index_raw = request.form.get("row_index", "").strip()
        if row_index_raw == "":
            error = "Record index missing. Select a row first."
        else:
            try:
                row_index = int(row_index_raw)
            except ValueError:
                row_index = -1
            if row_index < 0 or row_index >= len(df):
                error = "Invalid record index."
            else:
                df = df.drop(index=row_index).reset_index(drop=True)
                df.to_csv(dataset_path, index=False)
                message = f"Record deleted successfully from {dataset_path.name}."
                total_rows = len(df)
                if total_rows > 0:
                    next_index = min(row_index, total_rows - 1)
                    updated = df.loc[next_index, editable_columns].to_dict()
                    record = {
                        "row_index": next_index,
                        "values": {k: "" if pd.isna(v) else str(v) for k, v in updated.items()},
                    }
                else:
                    record = None

    return render_template(
        "records.html",
        username=session.get("username", "User"),
        active_page="records",
        datasets=list(datasets.keys()),
        dataset_key=dataset_key,
        record=record,
        row_index_query=row_index_query,
        total_rows=total_rows,
        message=message,
        error=error,
        embedded=embedded,
    )


@app.route("/graphs", methods=["GET"])
@login_required
def graphs():
    chart_path = TRAIN_PATH if TRAIN_PATH.exists() else DATA_PATH
    charts = {
        "risk_labels": ["Low Risk", "High Risk"],
        "risk_counts": [0, 0],
        "department_labels": [],
        "department_risk": [],
        "gender_labels": [],
        "gender_counts": [],
        "salary_points": [],
    }

    if chart_path.exists():
        df = pd.read_csv(chart_path)
        if "layoff_target" in df.columns:
            risk_counts = df["layoff_target"].astype(int).value_counts().reindex([0, 1], fill_value=0)
            charts["risk_counts"] = [int(risk_counts.loc[0]), int(risk_counts.loc[1])]

            if "department" in df.columns:
                dept_risk = (
                    df.groupby("department")["layoff_target"]
                    .mean()
                    .mul(100)
                    .sort_values(ascending=False)
                )
                charts["department_labels"] = dept_risk.index.astype(str).tolist()
                charts["department_risk"] = [round(float(v), 2) for v in dept_risk.tolist()]

            if "Gender" in df.columns:
                gender_counts = df["Gender"].astype(str).value_counts()
                charts["gender_labels"] = gender_counts.index.tolist()
                charts["gender_counts"] = [int(v) for v in gender_counts.tolist()]

            if "monthly_income" in df.columns:
                salary_df = df[["monthly_income", "layoff_target"]].dropna().copy()
                salary_df["monthly_income"] = pd.to_numeric(salary_df["monthly_income"], errors="coerce")
                salary_df = salary_df.dropna().sort_values("monthly_income").head(250)
                charts["salary_points"] = [
                    {"x": round(float(row["monthly_income"]), 2), "y": int(row["layoff_target"])}
                    for _, row in salary_df.iterrows()
                ]

    return render_template(
        "graphs.html",
        username=session.get("username", "User"),
        embedded=request.args.get("embed") == "1",
        charts=charts,
    )


@app.route("/download-template", methods=["GET"])
@login_required
def download_template():
    cols = meta["features"]
    template_df = pd.DataFrame([{c: "" for c in cols}])
    preview = template_df.to_html(index=False)
    return f"<h3>Template Columns</h3>{preview}"


if __name__ == "__main__":
    load_assets()
    app.run(debug=True)
