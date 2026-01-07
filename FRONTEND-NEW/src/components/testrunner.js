
import React, { useState } from "react";

const API_BASE = "http://localhost:8001";

export default function TestRunner() {
  const [loading, setLoading] = useState(false);
  const [reportLoading, setReportLoading] = useState(false);
  const [error, setError] = useState(null);

  const runTests = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/tests/run`, { method: "GET" });
      if (!res.ok) throw new Error(`Request failed: ${res.status}`);
      const data = await res.json();
      if (!data.report_url) throw new Error("No report_url returned");
      const url = data.report_url.startsWith("http") ? data.report_url : `${API_BASE}${data.report_url}`;
      window.open(url, "_blank", "noopener,noreferrer");
    } catch (e) {
      setError(e.message || "Failed to run tests");
    } finally {
      setLoading(false);
    }
  };

  const viewReport = async () => {
    setReportLoading(true);
    setError(null);
    try {
      // request report specifically for test_1.py
      const res = await fetch(`${API_BASE}/tests/report?test=tests/test_1.py`, { method: "GET" });
      if (!res.ok) throw new Error(`Request failed: ${res.status}`);
      const data = await res.json();
      if (!data.report_url) throw new Error("No report_url returned");
      const url = data.report_url.startsWith("http") ? data.report_url : `${API_BASE}${data.report_url}`;
      window.open(url, "_blank", "noopener,noreferrer");
    } catch (e) {
      setError(e.message || "Failed to fetch report");
    } finally {
      setReportLoading(false);
    }
  };

  return (
    <div style={{ padding: "1rem" }}>
      <h2>Test Runner</h2>
      <button onClick={viewReport} disabled={reportLoading} style={{ marginRight: "0.5rem" }}>
        {reportLoading ? "Preparing report…" : "Report"}
      </button>
      <button onClick={runTests} disabled={loading}>
        {loading ? "Running tests…" : "Run Tests & View Report"}
      </button>
      {error && <div style={{ color: "red", marginTop: "0.5rem" }}>{error}</div>}
    </div>
  );
}
