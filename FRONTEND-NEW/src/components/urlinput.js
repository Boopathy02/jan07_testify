import React, { useState } from "react";
import axios from "axios";
import API_BASE_URL from "../config";
import { toast } from "react-toastify";
import styles from "../css/URLInput.module.css";

const URLInput = ({ onBack, onNext, apiMode = "ocr" }) => {
  const [url, setUrl] = useState("");
  const [fullTestData, setFullTestData] = useState(null);
  const [loadingEnrich, setLoadingEnrich] = useState(false);
  const [loadingManualEnrich, setLoadingManualEnrich] = useState(false);
  const [error, setError] = useState("");

  const validateUrl = () => {
    if (!url || url.trim() === "") {
      setError("Please enter a valid URL");
      return false;
    }

    try {
      new URL(url);
    } catch (_) {
      setError("Please enter a valid URL format (e.g., https://example.com)");
      return false;
    }

    return true;
  };

  const enrichLocaters = async () => {
    if (!validateUrl()) {
      return;
    }

    setLoadingEnrich(true);
    setError("");
    setFullTestData(null);

    try {
      const endpoint = apiMode === "url" ? "/url/launch-browser" : "/launch-browser";
      const response = await axios.post(`${API_BASE_URL}${endpoint}`, {
        url: url
      });

      const data = response.data;
      setFullTestData(data);
      toast.success("Locators enriched successfully");
    } catch (err) {
      setError(err.response?.data?.message || "Error enriching locators");
    } finally {
      setLoadingEnrich(false);
    }
  };

  const manualEnrich = async () => {
    if (!validateUrl()) {
      return;
    }

    setLoadingManualEnrich(true);
    setError("");
    setFullTestData(null);

    try {
      const response = await axios.post(`${API_BASE_URL}/manual/launch-browser`, {
        url: url
      });

      toast.success(response.data?.message || "Manual enrich started");
    } catch (err) {
      setError(err.response?.data?.message || "Error starting manual enrich");
    } finally {
      setLoadingManualEnrich(false);
    }
  };

  return (
    <div className={styles.urlInputContainer}>
      <div className={styles.contentBox}>
        <h3 className={styles.title}>Enter URL</h3>
        <input
          type="text"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          placeholder="Paste your app URL here..."
          className={styles.urlInput}
        />
        {error && <p className={styles.errorText}>{error}</p>}

        <div className={styles.enrichButtonContainer}>
          <button
            onClick={enrichLocaters}
            className={styles.enrichButton}
          >
            {loadingEnrich ? "Enriching..." : "Enrich Locaters"}
          </button>
          {apiMode === "url" && (
            <button
              onClick={manualEnrich}
              className={styles.refreshButton}
            >
              {loadingManualEnrich ? "Launching..." : "Manual Enrich"}
            </button>
          )}
        </div>

        {fullTestData && (
          <div className={styles.testCaseOutput}>
            <h3 className={styles.testCaseOutputTitle}>
              Test Case JSON Output :
            </h3>
            <pre className={styles.jsonPre}>
              {JSON.stringify(fullTestData, null, 2)}
            </pre>
          </div>
        )}
      </div>

      <div className={styles.navigationButtons}>
        <button
          onClick={onBack}
          className={styles.navButton}
        >
          <i className="fa-solid fa-angle-left"></i>
          Previous
        </button>

        <button
          onClick={onNext}
          className={`${styles.navButton} ${styles.next}`}
        >
          Next <i className="fa-solid fa-angle-right"></i>
        </button>
      </div>
    </div>
  );
};

export default URLInput;

