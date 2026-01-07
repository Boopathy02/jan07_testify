import React, { useMemo, useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import axios from "axios";
import API_BASE_URL from "../config";
import { toast } from "react-toastify";
import styles from "../css/Execute.module.css";


const QualitySparkline = ({ data = [] }) => {
  const points = useMemo(() => {
    if (!data.length) return "";
    const values = data.map((item) => item.pass_rate ?? 0);
    const chartValues = values.length === 1 ? [values[0], values[0]] : values;
    const max = Math.max(...chartValues);
    const min = Math.min(...chartValues);
    const span = max - min || 1;
    return chartValues
      .map((value, index) => {
        const x = (index / (chartValues.length - 1 || 1)) * 100;
        const y = 100 - ((value - min) / span) * 100;
        return `${x},${y}`;
      })
      .join(" ");
  }, [data]);

  if (!points) {
    return <div className={styles.metricsSparklinePlaceholder}>No data</div>;
  }

  return (
    <svg viewBox="0 0 100 100" className={styles.metricsSparkline}>
      <polyline
        points={points}
        fill="none"
        stroke="var(--color-primary)"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
};

const Execute = ({ onBack, fullTestData }) => {
  const navigate = useNavigate();
  const [loadingExecution, setLoadingExecution] = useState(false);
  const [runningTestName, setRunningTestName] = useState("");
  const [executionSuccess, setExecutionSuccess] = useState(false);
  const [executionError, setExecutionError] = useState(false);
  const [error, setError] = useState("");

  const [reportLoading, setReportLoading] = useState(false);
  const [visualizerImages, setVisualizerImages] = useState([]);
  const [visualizerLoading, setVisualizerLoading] = useState(false);
  const [visualizerError, setVisualizerError] = useState("");
  const [showVisualizer, setShowVisualizer] = useState(false);
  const [visualizerMode, setVisualizerMode] = useState("idle"); // idle | interactive | images
  const [visualizerDashboardUrl, setVisualizerDashboardUrl] = useState("");

  const [metrics, setMetrics] = useState(null);
  const [metricsLoading, setMetricsLoading] = useState(false);
  const [metricsError, setMetricsError] = useState("");
  const [plannedTests, setPlannedTests] = useState([]);
  const [plannedTestsProjectId, setPlannedTestsProjectId] = useState(null);
  const [plannedTestsProjectName, setPlannedTestsProjectName] = useState("");
  const [executionFilters, setExecutionFilters] = useState({
    ui: { regression: false, functional: false },
    accessibility: { regression: false, functional: false },
    security: { regression: false, functional: false },
  });
  const [categorySelections, setCategorySelections] = useState({
    accessibility: false,
    security: false,
  });
  const [useUpdatedExecutionByCategory, setUseUpdatedExecutionByCategory] = useState({
    ui: false,
    accessibility: false,
    security: false,
  });
  const [tagCounts, setTagCounts] = useState({});
  const [filtersExpanded, setFiltersExpanded] = useState({
    ui: true,
    accessibility: false,
    security: false,
  });
  const [testPageIndex, setTestPageIndex] = useState(1);
  const testsPerPage = 8;

  const hasSelectedTags = useMemo(() => {
    const uiSelected = Object.values(executionFilters.ui).some(Boolean);
    return uiSelected || categorySelections.accessibility || categorySelections.security;
  }, [executionFilters, categorySelections]);

  const hasUpdatedSelection = useMemo(
    () => Object.values(useUpdatedExecutionByCategory).some(Boolean),
    [useUpdatedExecutionByCategory]
  );

  const formatServerError = (err, fallbackMessage) => {
    const detail =
      err?.response?.data?.detail ??
      err?.response?.data?.message ??
      err?.response?.data ??
      err?.message;
    if (typeof detail === "object") {
      try {
        return JSON.stringify(detail);
      } catch {
        return fallbackMessage;
      }
    }
    return detail || fallbackMessage;
  };

  useEffect(() => {
    if (!hasSelectedTags && !hasUpdatedSelection) {
      setPlannedTests([]);
      setPlannedTestsProjectId(null);
      setPlannedTestsProjectName("");
      setTagCounts({});
      setTestPageIndex(1);
      return;
    }

    const timer = setTimeout(async () => {
      try {
        const selectedTags = {
          ui: Object.keys(executionFilters.ui).filter((tag) => executionFilters.ui[tag]),
          accessibility: categorySelections.accessibility ? ["__all__"] : [],
          security: categorySelections.security ? ["__all__"] : [],
        };
        const res = await axios.post(`${API_BASE_URL}/rag/preview-tests`, {
          tags: selectedTags,
          use_test_plan: hasUpdatedSelection ? useUpdatedExecutionByCategory : false,
        });
        const data = res.data || {};
        if (Array.isArray(data.planned_tests)) {
          setPlannedTests(data.planned_tests);
          setTestPageIndex(1);
        }
        if (data.project_id) {
          setPlannedTestsProjectId(data.project_id);
        }
        if (data.project_name) {
          setPlannedTestsProjectName(data.project_name);
        }
        if (data.tag_counts) {
          setTagCounts(data.tag_counts);
        } else {
          setTagCounts({});
        }
      } catch (err) {
        setError(formatServerError(err, "Failed to preview planned tests."));
      }
    }, 250);

    return () => clearTimeout(timer);
  }, [executionFilters, categorySelections, useUpdatedExecutionByCategory, hasSelectedTags, hasUpdatedSelection]);

  const executeStoryTest = async (testsToRun = null, forcedCategory = null) => {
    const hasSingleTest = Array.isArray(testsToRun) && testsToRun.length > 0;
    const plannedRunNames =
      !hasSingleTest && flatPlannedTests.length
        ? Array.from(new Set(flatPlannedTests.map((item) => item.testName)))
        : [];
    setLoadingExecution(true);
    setError("");
    setPlannedTests([]);
    setPlannedTestsProjectId(null);
    setPlannedTestsProjectName("");
    setTestPageIndex(1);
    setExecutionSuccess(false);
    setExecutionError(false);
    if (hasSingleTest) {
      setRunningTestName(testsToRun[0]);
    }

    try {
      // Calls the correct RAG runner endpoint using POST
      const baseTags = {
        ui: Object.keys(executionFilters.ui).filter((tag) => executionFilters.ui[tag]),
        accessibility: categorySelections.accessibility ? ["__all__"] : [],
        security: categorySelections.security ? ["__all__"] : [],
      };
      const selectedTags = forcedCategory
        ? {
            ui: [],
            accessibility: [],
            security: [],
            [forcedCategory]: ["__all__"],
          }
        : baseTags;
      const useUpdated = hasSingleTest || plannedRunNames.length
        ? false
        : Object.values(useUpdatedExecutionByCategory).some(Boolean);
      const payload = {
        tags: selectedTags,
        use_test_plan: useUpdated ? useUpdatedExecutionByCategory : false,
      };
      if (hasSingleTest) {
        payload.tests_to_run = testsToRun;
      } else if (plannedRunNames.length) {
        payload.tests_to_run = plannedRunNames;
      }
      const res = await axios.post(`${API_BASE_URL}/rag/run-generated-story-test`, payload);
      const data = res.data;

      if (!data) {
        throw new Error("No response data returned from server");
      }

      if (Array.isArray(data.planned_tests)) {
        setPlannedTests(data.planned_tests);
        setTestPageIndex(1);
      }
      if (Array.isArray(data.planned_tests_to_run) && data.planned_tests_to_run.length) {
        toast.info(`Planned tests to run: ${data.planned_tests_to_run.join(", ")}`);
      }
      if (data.project_id) {
        setPlannedTestsProjectId(data.project_id);
      }
      if (data.project_name) {
        setPlannedTestsProjectName(data.project_name);
      }

      if (data.status === "PASS") {
        if (hasSingleTest) {
          toast.success(`Test executed successfully: ${testsToRun[0]}`);
        } else {
          toast.success(`✅ Test executed successfully: ${data.executed_from}`);
        }
        setExecutionSuccess(true);
      } else {
        toast.error(`❌ Test failed. Check logs for details.`);
        setExecutionError(true);
      }
      
      // After execution, refresh the metrics dashboard
      await fetchMetrics();

    } catch (err) {
      const formattedError = formatServerError(err, "Error executing the test script.");
      setError(formattedError);
      setExecutionError(true);
      toast.error(formattedError);
    } finally {
      setLoadingExecution(false);
      setRunningTestName("");
    }
  };

  const viewReport = async () => {
    setReportLoading(true);
    setError("");
    setPlannedTests([]);
    try {
      const res = await axios.get(`${API_BASE_URL}/tests/open`, { params: { test: "tests/test_1.py" } });
      const data = res.data;
      if (!data) throw new Error("No response data returned from server");
      const uri = data.report_url || data.report_uri || data.file_uri || data.path;
      if (!uri) throw new Error("No report URL/URI returned from server");
      window.open(uri, "_blank", "noopener,noreferrer");
    } catch (err) {
      setError(formatServerError(err, "Failed to open report"));
    } finally {
      setReportLoading(false);
    }
  };

  const formatVisualizerAssetUrl = (url) => {
    if (!url) return "";
    if (/^https?:\/\//i.test(url)) {
      return url;
    }
    const base = API_BASE_URL.endsWith("/") ? API_BASE_URL.slice(0, -1) : API_BASE_URL;
    const path = url.startsWith("/") ? url : `/${url}`;
    return `${base}${path}`;
  };

  const loadVisualizerContent = async () => {
    if (visualizerLoading) {
      return;
    }
    setVisualizerLoading(true);
    setVisualizerError("");
    try {
      const res = await axios.get(`${API_BASE_URL}/visualizer/images`);
      const images = Array.isArray(res.data?.images) ? res.data.images : [];
      const dashboardPath = res.data?.interactive_dashboard;

      if (dashboardPath) {
        const absoluteDashboard = formatVisualizerAssetUrl(dashboardPath);
        setVisualizerDashboardUrl(`${absoluteDashboard}?t=${Date.now()}`);
        setVisualizerMode("interactive");
        setVisualizerImages([]);
      } else {
        setVisualizerImages(images);
        setVisualizerMode("images");
      }

      setShowVisualizer(true);
    } catch (err) {
      setVisualizerError(formatServerError(err, "Failed to load visualizations."));
    } finally {
      setVisualizerLoading(false);
    }
  };

  const closeVisualizer = () => {
    setShowVisualizer(false);
    setVisualizerError("");
  };

  const handleToggleVisualizer = async () => {
    if (visualizerLoading) {
      return;
    }
    if (showVisualizer) {
      closeVisualizer();
      return;
    }

    if (visualizerMode === "idle") {
      await loadVisualizerContent();
      return;
    }

    setShowVisualizer(true);
  };

  const handleRefreshVisualizer = async () => {
    if (visualizerLoading) {
      return;
    }
    if (visualizerMode === "interactive" && visualizerDashboardUrl) {
      const baseUrl = visualizerDashboardUrl.split("?")[0];
      setVisualizerDashboardUrl(`${baseUrl}?t=${Date.now()}`);
      return;
    }
    await loadVisualizerContent();
  };

  const formatPercent = (value) => `${Math.round((value ?? 0) * 100)}%`;
  const formatCount = (value) => (value ?? 0);
  const summary = metrics?.self_healing_summary || {};
  const periods = metrics?.periods || {};
  const latestStatus = metrics?.latest_run?.status_counts || {};
  const selfHealing = metrics?.self_healing_reports || {};
  const healingStrategies = selfHealing.strategy_usage || [];
  const healingSteps = selfHealing.healing_steps_per_feature || [];
  const healingHistory = selfHealing.history || [];

  const fetchMetrics = async () => {
    setMetricsLoading(true);
    setMetricsError("");
    try {
      const res = await axios.get(`${API_BASE_URL}/metrics/dashboard`);
      setMetrics(res.data);
      } catch (err) {
      setMetricsError(formatServerError(err, "Unable to load quality metrics."));
    } finally {
      setMetricsLoading(false);
    }
  };

  useEffect(() => {
    fetchMetrics();
  }, []);

  const toggleExecutionTag = (category, tag) => {
    setExecutionFilters((prev) => ({
      ...prev,
      [category]: {
        ...prev[category],
        [tag]: !prev[category][tag],
      },
    }));
  };

  const toggleFilterSection = (category) => {
    setFiltersExpanded((prev) => ({
      ...prev,
      [category]: !prev[category],
    }));
  };

  const toggleUpdatedExecution = (category) => {
    setUseUpdatedExecutionByCategory((prev) => ({
      ...prev,
      [category]: !prev[category],
    }));
  };

  const handleOpenTestFile = (filePath) => {
    if (!plannedTestsProjectId || !filePath) {
      return;
    }
    navigate("/editor", {
      state: {
        openFile: {
          projectId: plannedTestsProjectId,
          projectName: plannedTestsProjectName,
          path: filePath,
        },
      },
    });
  };

  const flatPlannedTests = useMemo(() => {
    if (!plannedTests.length) {
      return [];
    }
    const flattened = [];
      plannedTests.forEach((plan) => {
        const tests = plan.tests || [];
        const category = plan.category || "ui";
        const scriptKey = plan.script || plan.script_path || plan.name || "";
        tests.forEach((testName) => {
          const filePath =
            plan.test_files?.find((item) => item.name === testName)?.path || plan.script_path;
          flattened.push({
            category,
            scriptKey,
          script_path: plan.script_path,
          test_files: plan.test_files,
          testName,
          filePath,
        });
      });
    });
    return flattened;
  }, [plannedTests]);

  const totalPlannedCount = flatPlannedTests.length;
  const totalPages = Math.max(1, Math.ceil(totalPlannedCount / testsPerPage));
  const pagedPlans = useMemo(() => {
    if (!flatPlannedTests.length) {
      return [];
    }
    const start = (testPageIndex - 1) * testsPerPage;
    const pageItems = flatPlannedTests.slice(start, start + testsPerPage);
    const grouped = new Map();
    pageItems.forEach((item) => {
      const key = `${item.category}:${item.scriptKey}`;
      if (!grouped.has(key)) {
        grouped.set(key, {
          category: item.category,
          script: item.scriptKey,
          script_path: item.script_path,
          test_files: item.test_files,
          tests: [],
        });
      }
      grouped.get(key).tests.push({ name: item.testName, path: item.filePath });
    });
    return Array.from(grouped.values());
  }, [flatPlannedTests, testPageIndex, testsPerPage]);

  return (
    <div className={styles.executeContainer}>
      <div className={styles.contentBox}>
        {/* Heading Section */}
        <h3 className={styles.heading}>
          <i className={`fa-solid fa-code ${styles.headingIcon}`}></i>
          Generate Scripts
        </h3>
        <p className={styles.subheading}>Configure framework and generate test scripts</p>

        {/* Icon & Description */}
        <div className={styles.centerContent}>
          <div className={styles.mainIcon}>
            <i className="fa-solid fa-code"></i>
          </div>
          <h2 className={styles.mainTitle}>Generate Test Scripts</h2>
          <p className={styles.mainDescription}>
            Your test scripts will be generated based on the uploaded designs and user stories.
          </p>
        </div>

        <div className={styles.metricsWrapper}>
          <div className={styles.metricsHeader}>
            <div>
              <h4>Quality &amp; Self-Healing Insights</h4>
              <p>Pass rates, self-healing power, flakiness, and risk heatmap generated from Allure.</p>
            </div>
            {metricsLoading && <span className={styles.metricsLoading}>Loading metrics…</span>}
          </div>

          {metricsError && <p className={styles.metricsError}>{metricsError}</p>}

          {metrics && (
            <>
              <div className={styles.metricCards}>
                <div className={styles.metricCard}>
                  <span className={styles.metricLabel}>Self-healing save rate</span>
                  <strong>{formatPercent(summary.save_rate)}</strong>
                  <p>
                    {summary.saved ?? 0}/{summary.tests ?? 0} tests healed successfully.
                  </p>
                </div>
                <div className={styles.metricCard}>
                  <span className={styles.metricLabel}>Self-healing power</span>
                  <strong>{formatPercent(summary.power)}</strong>
                  <p>Pass rate × save rate indicates stability improvement.</p>
                </div>
                <div className={styles.metricCard}>
                  <span className={styles.metricLabel}>Last run status</span>
                  <strong>
                    {formatCount(latestStatus.passed)} - pass  
                    <br/>{formatCount(latestStatus.failed)} - fail {" "}
                    <br/>{formatCount(latestStatus.broken)} - broken
                  </strong>
                  <p>{summary.total ?? 0} tests executed</p>
                </div>
                <div className={styles.metricCard}>
                  <span className={styles.metricLabel}>Trend window</span>
                  <strong>
                    {formatCount(periods?.["7_days"]?.passed)} - passed this week {" "}
                    <br/>{formatCount(periods?.["7_days"]?.failed)} - failed
                  </strong>
                  <p>{formatCount(periods?.["30_days"]?.total)} tests last 30 days</p>
                </div>
              </div>

              <div className={styles.metricsCharts}>
                <div className={styles.metricsChart}>
                  <div className={styles.chartHeader}>
                    <h5>Quality over time</h5>
                    <span>Pass rate history</span>
                  </div>
                  <QualitySparkline data={metrics.pass_rate_history} />
                </div>
                <div className={styles.metricsChart}>
                  <div className={styles.chartHeader}>
                    <h5>Self-healed actions</h5>
                    <span>Last {metrics.healing_history?.length ?? 0} runs</span>
                  </div>
                  <ul className={styles.metricsHistory}>
                    {(metrics.healing_history || []).slice(-5).map((item, index) => {
                      const label = item.timestamp ? new Date(item.timestamp).toLocaleString() : "Recent run";
                      return (
                        <li key={`${item.timestamp ?? "healing"}-${index}`}>
                          <strong>{item.healing_actions ?? 0}</strong> actions ·{" "}
                          <span>{item.healing_saved ?? 0} saved</span>
                          <small>{label}</small>
                        </li>
                      );
                    })}
                  </ul>
                </div>
              </div>

              <div className={styles.metricsLists}>
                <div className={styles.metricsList}>
                  <h5>Flaky Champions</h5>
                  <ul>
                    {(metrics.flaky_champions || []).slice(0, 5).map((flaky) => (
                      <li key={flaky.name}>
                        <span>{flaky.name}</span>
                        <small>
                          {flaky.flaky_score} flips · {flaky.failures} failures · {flaky.healing_actions} healing
                        </small>
                      </li>
                    ))}
                  </ul>
                </div>
                <div className={styles.metricsList}>
                  <h5>Risk heatmap / top healed pages</h5>
                  <ul>
                    {(metrics.risk_heatmap || []).slice(0, 5).map((feature) => (
                      <li key={feature.name}>
                        <span>{feature.name}</span>
                        <small>
                          fail rate {formatPercent(feature.fail_rate)} · {feature.healing_actions} healing
                        </small>
                      </li>
                    ))}
                  </ul>
                  <div className={styles.topHealed}>
                    <strong>Top healed pages</strong>
                    <ul>
                      {(metrics.top_healed_pages || []).map((page) => (
                        <li key={`healed-${page.name}`}>
                          <span>{page.name}</span>
                          <small>{page.healing_actions} healing actions</small>
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
              </div>
              <div className={styles.selfHealingSection}>
                <div className={styles.selfHealingHeader}>
                  <h5>Self-Healing Intelligence</h5>
                  <span>SmartAI actions per run</span>
                </div>
                <div className={styles.selfHealingCards}>
                  <div className={styles.selfHealingCard}>
                    <span className={styles.metricLabel}>Saved failures</span>
                    <strong>{formatCount(summary.saved)}</strong>
                    <p>{formatPercent(summary.save_rate)} save rate</p>
                  </div>
                  <div className={styles.selfHealingCard}>
                    <span className={styles.metricLabel}>Healing actions</span>
                    <strong>{formatCount(summary.actions)}</strong>
                    <p>{formatPercent(summary.power)} heal power</p>
                  </div>
                  <div className={styles.selfHealingCard}>
                    <span className={styles.metricLabel}>Failed heals</span>
                    <strong>{formatCount(selfHealing.summary?.failed || 0)}</strong>
                    <p>Need tuning</p>
                  </div>
                </div>
                <div className={styles.selfHealingGrid}>
                  <div className={styles.selfHealingList}>
                    <h6>Strategy usage</h6>
                    <ul>
                      {healingStrategies.length ? (
                        healingStrategies.slice(0, 5).map((strategy) => (
                          <li key={strategy.strategy}>
                            <span>{strategy.strategy}</span>
                            <small>{formatCount(strategy.count)} uses</small>
                          </li>
                        ))
                      ) : (
                        <li>No healing strategy data yet.</li>
                      )}
                    </ul>
                  </div>
                  <div className={styles.selfHealingList}>
                    <h6>Healed steps by feature</h6>
                    <ul>
                      {healingSteps.length ? (
                        healingSteps.slice(0, 5).map((entry) => (
                          <li key={entry.name}>
                            <span>{entry.name}</span>
                            <small>
                              {formatCount(entry.healed_steps)} of {formatCount(entry.total_steps)} steps healed
                            </small>
                          </li>
                        ))
                      ) : (
                        <li>Run tests with healing to collect data.</li>
                      )}
                    </ul>
                  </div>
                  <div className={styles.selfHealingList}>
                    <h6>Recent healing runs</h6>
                    <ul>
                      {healingHistory.length ? (
                        healingHistory.slice(-3).map((item) => (
                          <li key={item.timestamp}>
                            <span>{formatCount(item.healing_actions)} actions</span>
                            <small>{new Date(item.timestamp).toLocaleString()}</small>
                          </li>
                        ))
                      ) : (
                        <li>No history yet.</li>
                      )}
                    </ul>
                  </div>
                </div>
              </div>
            </>
          )}
        </div>

        {/* Action Buttons: Report + Execute + Visualize */}
        <div className={styles.executeButtonContainer}>
          <div className={styles.tagFilterBox}>
            <div className={styles.tagFilterTitle}>Execution filters</div>
            <div className={styles.tagDropdownList}>
              {[
                { id: "ui", label: "UI" },
                { id: "accessibility", label: "Accessibility" },
                { id: "security", label: "Security" },
              ].map((section) => (
                <div key={section.id} className={styles.tagDropdown}>
                  <button
                    type="button"
                    className={styles.tagDropdownToggle}
                    onClick={() => toggleFilterSection(section.id)}
                  >
                    <span>{section.label}</span>
                    <i
                      className={`fa-solid fa-chevron-${filtersExpanded[section.id] ? "up" : "down"}`}
                      aria-hidden="true"
                    />
                  </button>
                  {filtersExpanded[section.id] && (
                    <>
                      {section.id === "ui" ? (
                        <>
                          <div className={styles.tagFilterGrid}>
                            {["regression", "functional"].map((tag) => {
                              const count = tagCounts?.[section.id]?.[tag] ?? 0;
                              return (
                                <label key={`${section.id}-${tag}`} className={styles.tagFilterItem}>
                                  <input
                                    type="checkbox"
                                    className={styles.tagCheckbox}
                                    checked={executionFilters[section.id][tag]}
                                    onChange={() => toggleExecutionTag(section.id, tag)}
                                  />
                                  <span>
                                    {tag.charAt(0).toUpperCase() + tag.slice(1)}
                                    {executionFilters[section.id][tag] ? ` (${count})` : ""}
                                  </span>
                                </label>
                              );
                            })}
                          </div>
                          <label className={styles.executionPlanToggle}>
                            <input
                              type="checkbox"
                              checked={useUpdatedExecutionByCategory[section.id]}
                              onChange={() => toggleUpdatedExecution(section.id)}
                            />
                            Run only updated tests (from last changes)
                          </label>
                        </>
                      ) : (
                        <label className={styles.executionPlanToggle}>
                          <input
                            type="checkbox"
                            checked={categorySelections[section.id]}
                            onChange={() =>
                              setCategorySelections((prev) => ({
                                ...prev,
                                [section.id]: !prev[section.id],
                              }))
                            }
                          />
                          Include {section.label} tests
                          {categorySelections[section.id]
                            ? ` (${plannedTests
                                .filter((plan) => plan.category === section.id)
                                .reduce((total, plan) => total + (plan.tests?.length || 0), 0)})`
                            : ""}
                        </label>
                      )}
                    </>
                  )}
                </div>
              ))}
            </div>
          </div>
          <div className={styles.actionButtons}>
            <button onClick={viewReport} disabled={true} className={styles.reportButton}>
              {reportLoading ? "Opening report..." : "Report"}
            </button>

            <button onClick={executeStoryTest} disabled={loadingExecution} className={styles.executeButton}>
              {loadingExecution ? "Executing..." : "Execute"}
            </button>

            <button onClick={handleToggleVisualizer} disabled={visualizerLoading} className={styles.visualizeButton}>
              {visualizerLoading ? "Loading visuals..." : showVisualizer ? "Hide Visual Charts" : "Visualize Charts"}
            </button>
          </div>
          {error && <div className={styles.errorLog}>{error}</div>}
          {plannedTests.length > 0 && (
            <div className={styles.executionPlan}>
              <div className={styles.executionPlanHeader}>
                <h4>Planned test cases</h4>
                <span>
                  {totalPlannedCount} selected
                </span>
              </div>
              {pagedPlans.map((plan) => (
                <div key={`${plan.category}-${plan.script}`} className={styles.executionPlanBlock}>
                  <strong>{(plan.category || "ui").toUpperCase()}</strong>
                    {plan.tests && plan.tests.length ? (
                      <ul>
                        {plan.tests.map((testItem) => {
                          const testName = testItem?.name ?? testItem;
                          const filePath = testItem?.path;
                            return (
                              <li key={`${plan.category}-${testName}`}>
                                <div className={styles.executionPlanItem}>
                                  {filePath ? (
                                    <button
                                      type="button"
                                      className={styles.executionPlanLink}
                                      onClick={() => handleOpenTestFile(filePath)}
                                    >
                                      {testName}
                                    </button>
                                  ) : (
                                    <span className={styles.executionPlanName}>{testName}</span>
                                  )}
                                  <button
                                    type="button"
                                    className={styles.executionPlanRunButton}
                                    onClick={() => executeStoryTest([testName], plan.category || "ui")}
                                    disabled={loadingExecution || runningTestName === testName}
                                  >
                                    {runningTestName === testName ? "Executing..." : "Execute"}
                                  </button>
                                </div>
                              </li>
                            );
                          })}
                      </ul>
                    ) : (
                    <p>No matching tests for the selected tags.</p>
                  )}
                </div>
              ))}
              {totalPages > 1 && (
                <div className={styles.executionPagination}>
                  <button
                    type="button"
                    className={styles.executionPageButton}
                    onClick={() => setTestPageIndex((prev) => Math.max(1, prev - 1))}
                    disabled={testPageIndex === 1}
                  >
                    Prev
                  </button>
                  <span className={styles.executionPageInfo}>
                    Page {testPageIndex} of {totalPages}
                  </span>
                  <button
                    type="button"
                    className={styles.executionPageButton}
                    onClick={() => setTestPageIndex((prev) => Math.min(totalPages, prev + 1))}
                    disabled={testPageIndex === totalPages}
                  >
                    Next
                  </button>
                </div>
              )}
            </div>
          )}

          {showVisualizer && (
            <div className={styles.visualizerPanel}>
              <div className={styles.visualizerPanelHeader}>
                <h4>Allure Visualizations</h4>
                <button type="button" onClick={closeVisualizer} className={styles.visualizerPanelClose}>
                  Close
                </button>
              </div>
              {visualizerLoading && (
                <p className={styles.visualizerStatus}>Fetching the latest visualizations…</p>
              )}
              {visualizerError && <p className={styles.visualizerError}>{visualizerError}</p>}
              {!visualizerLoading && !visualizerError && (
                <div className={styles.visualizerToolbar}>
                  <span>
                    {visualizerMode === "interactive"
                      ? "Interactive Plotly dashboard"
                      : visualizerImages.length
                      ? "Static chart snapshots"
                      : "No visualizations detected yet"}
                  </span>
                  <div className={styles.visualizerToolbarActions}>
                    <button type="button" onClick={handleRefreshVisualizer}>
                      Refresh
                    </button>
                    {visualizerMode === "interactive" && visualizerDashboardUrl && (
                      <a
                        href={visualizerDashboardUrl}
                        target="_blank"
                        rel="noopener noreferrer"
                      >
                        Open full screen
                      </a>
                    )}
                  </div>
                </div>
              )}
              {!visualizerLoading && !visualizerError && (
                visualizerMode === "interactive" && visualizerDashboardUrl ? (
                  <div className={styles.visualizerIframeWrapper}>
                    <iframe
                      key={visualizerDashboardUrl}
                      src={visualizerDashboardUrl}
                      title="Interactive Allure dashboard"
                      className={styles.visualizerIframe}
                      loading="lazy"
                    />
                  </div>
                ) : visualizerImages.length ? (
                  <div className={styles.visualizerGrid}>
                    {visualizerImages.map((image) => (
                      <div key={image.name} className={styles.visualizerCard}>
                        <img
                          src={formatVisualizerAssetUrl(image.url)}
                          alt={image.name}
                          className={styles.visualizerImage}
                        />
                        <span className={styles.visualizerCaption}>{image.name}</span>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className={styles.visualizerStatus}>
                    No visualizations found yet. Run the Allure visualizer or execute tests to generate charts under
                    the backend's allure_reports folder.
                  </p>
                )
              )}
            </div>
          )}
        </div>
      </div>

      {/* Back Button */}
      <div className={styles.backButtonContainer}>
        <button onClick={onBack} className={styles.backButton}>
          <i className="fa-solid fa-angle-left"></i>
          Back
        </button>
      </div>
    </div>
  );
};

export default Execute;
