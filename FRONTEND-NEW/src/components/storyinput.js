import React, { useState } from "react";
import axios from "axios";
import * as XLSX from "xlsx";
import { toast, ToastContainer } from "react-toastify";
import styles from "../css/StoryInput.module.css";
import API_BASE_URL from "../config";

const TEST_TYPE_OPTIONS = [
  { label: "UI Tests", value: "ui" },
  { label: "Security Tests", value: "security" },
  { label: "Accessibility Tests", value: "accessibility" },
];

const StoryInput = ({ onBack, onNext, testCases, setTestCases }) => {
  const [userStoriesInput, setUserStoriesInput] = useState("");
  const [selectedFile, setSelectedFile] = useState(null);
  const [selectedTestTypes, setSelectedTestTypes] = useState([]);

  const [loadingGeneration, setLoadingGeneration] = useState(false);
  const [loadingJira, setLoadingJira] = useState(false);
  const [loadingExcel, setLoadingExcel] = useState(false);
  const [error, setError] = useState("");

  const [showGitModal, setShowGitModal] = useState(false);
  const [repoUrl, setRepoUrl] = useState("");
  const [branchName, setBranchName] = useState("main");
  const [commitMessage, setCommitMessage] = useState("Add automated test cases");
  const [isPushingToGit, setIsPushingToGit] = useState(false);

  const handleTestTypeChange = (e) => {
    const { value, checked } = e.target;
    setSelectedTestTypes((prev) => {
      if (checked) {
        return [...prev, value];
      } else {
        return prev.filter((type) => type !== value);
      }
    });
  };

  // Fetch test cases from backend
  const fetchTestCases = async () => {
    if ((!userStoriesInput || userStoriesInput.trim() === "") && !selectedFile) {
      setError("Please enter at least one user story or upload a file.");
      return;
    }

    if (selectedTestTypes.length === 0) {
      setError("Select at least one test type.");
      return;
    }

    try {
      setLoadingGeneration(true);
      setError("");

      if (selectedFile) {
        const aggregated = [];
        for (const type of selectedTestTypes) {
          const formData = new FormData();
          formData.append("file", selectedFile);
          formData.append("test_type", type);
          const res = await axios.post(`${API_BASE_URL}/rag/generate-from-story`, formData, {
            headers: { "Content-Type": "multipart/form-data" },
          });
          if (Array.isArray(res.data?.results)) {
            aggregated.push(...res.data.results);
          }
        }

        setTestCases(aggregated);
        toast.success("Test cases generated successfully.");
      } else {
        const stories = userStoriesInput
          .split("|")
          .map((s) => s.trim())
          .filter((s) => s.length > 0);

        if (stories.length === 0) {
          setError("Please enter at least one valid user story separated by | ");
          setLoadingGeneration(false);
          return;
        }

        const aggregated = [];
        for (const s of stories) {
          for (const type of selectedTestTypes) {
            const res = await axios.post(
              `${API_BASE_URL}/rag/generate-from-story`,
              new URLSearchParams({
                user_story: s,
                test_type: type,
              })
            );
            if (Array.isArray(res.data?.results)) {
              aggregated.push(...res.data.results);
            }
          }
        }

        setTestCases(aggregated);
        toast.success(`Generated ${aggregated.length} test case(s) for ${stories.length} stor${stories.length > 1 ? "ies" : "y"}.`);
      }
    } catch (err) {
      console.error(err);
      setError(err.response?.data?.detail || "Error generating test cases.");
    } finally {
      setLoadingGeneration(false);
    }
  };  // Import from Jira
  const handleJiraImport = async () => {
    setLoadingJira(true);
    try {
      const response = await axios.get(`${API_BASE_URL}/jira/import`);
      const importedStories = response.data?.stories || [];

      if (importedStories.length > 0) {
        // join with pipe so UI shows the delimiter clearly
        setUserStoriesInput(importedStories.join(" |\n"));
        setSelectedFile(null);
        toast.success("User stories imported from Jira.");
      } else {
        toast.info("No stories found in Jira.");
      }
    } catch (err) {
      console.error(err);
      toast.error("Failed to import stories from Jira.");
    } finally {
      setLoadingJira(false);
    }
  };

  // Import from Excel: extract 'User Story' from 'User Stories' sheet
  const handleExcelImport = () => {
    setLoadingExcel(true);
    const input = document.createElement("input");
    input.type = "file";
    input.accept = ".xlsx, .xls";

    input.onchange = (e) => {
      const file = e.target.files[0];
      if (!file) return;

      const reader = new FileReader();

      reader.onload = (event) => {
        try {
          const data = new Uint8Array(event.target.result);
          const workbook = XLSX.read(data, { type: "array" });

          const userStoriesSheet = workbook.Sheets["User Stories"];
          if (!userStoriesSheet) {
            toast.error("Sheet named 'User Stories' not found.");
            setLoadingExcel(false);
            return;
          }

          const jsonSheet = XLSX.utils.sheet_to_json(userStoriesSheet, { defval: "" });

          const userStoryColKey = jsonSheet.length
            ? Object.keys(jsonSheet[0]).find((k) => k.trim().toLowerCase() === "user story")
            : null;

          if (!userStoryColKey) {
            toast.error("Column 'User Story' not found in 'User Stories' sheet.");
            setLoadingExcel(false);
            return;
          }

          const stories = jsonSheet
            .map((row) => row[userStoryColKey])
            .filter((val) => typeof val === "string" && val.trim().length > 0);

          // Join with pipe so user can see delimiters; fetchTestCases will split on '|'
          setUserStoriesInput(stories.join(" |\n"));
          setSelectedFile(file);
          toast.success("User stories imported from Excel.");
        } catch (err) {
          console.error(err);
          toast.error("Failed to import user stories from Excel.");
        }
        setLoadingExcel(false);
      };

      reader.readAsArrayBuffer(file);
    };

    input.click();
  };

  // Git push feature functions
  const handlePushToGitClick = () => {
    setShowGitModal(true);
  };

  const handleGitModalSubmit = async () => {
    if (!repoUrl.trim() || !branchName.trim() || !commitMessage.trim()) {
      toast.error("All Git fields are required.");
      return;
    }

    setIsPushingToGit(true);
    try {
      const testCasesContent = testCases.map((tc) => tc.auto_testcase);

      const response = await axios.post(`${API_BASE_URL}/git/push-generated-runs`, {
        repo_url: repoUrl,
        branch_name: branchName,
        commit_message: commitMessage,
      });

      if (response.status === 200) {
        toast.success("Test cases pushed to Git successfully!");
        setShowGitModal(false);

        setRepoUrl("");
        setBranchName("main");
        setCommitMessage("Add automated test cases");
      } else {
        toast.error(`Failed to push to Git: ${response.data?.detail || "Unknown error"}`);
      }
    } catch (err) {
      console.error("Git push error:", err);
      toast.error(`Error pushing to Git: ${err.response?.data?.detail || err.message || "Please try again."}`);
    } finally {
      setIsPushingToGit(false);
    }
  };


  const handleGitModalClose = () => {
    setShowGitModal(false);
    setRepoUrl("");
    setBranchName("main");
    setCommitMessage("Add automated test cases");
  };

  return (
    <div className={styles.storyInputContainer}>
      <ToastContainer />
      <div className={styles.contentBox}>
        <h3 className={styles.title}>Import User Stories</h3>
        <p>Add user stories from Jira, Excel, or create them manually</p>

        <div className={styles.importOptions}>
          {/* Manual Entry */}
          <button
            onClick={() => {
              setSelectedFile(null);
            }}
            className={`${styles.optionCard} ${styles.clickable}`}
          >
            <i className={`fa-solid fa-plus ${styles.optionIcon}`}></i>
            <h3 className={styles.optionTitle}>Manual Entry</h3>
            <p className={styles.optionDescription}>Add user stories manually</p>
          </button>

          {/* Jira Import */}
          <button
            onClick={handleJiraImport}
            disabled={loadingJira}
            className={`${styles.optionCard} ${styles.clickable}`}
          >
            <i className={`fa-solid fa-file-import ${styles.optionIcon}`}></i>
            <h3 className={styles.optionTitle}>
              {loadingJira ? <div className={styles.spinner}></div> : "Import from Jira"}
            </h3>
            <p className={styles.optionDescription}>Connect to Jira Instance</p>
          </button>

          {/* Excel Import */}
          <button
            onClick={handleExcelImport}
            disabled={loadingExcel}
            className={`${styles.optionCard} ${styles.clickable}`}
          >
            <i className={`fa-solid fa-file ${styles.optionIcon}`}></i>
            <h3 className={styles.optionTitle}>
              {loadingExcel ? (
                <div className={styles.spinner}></div>
              ) : (
                "Import Excel" + (selectedFile ? ` (${selectedFile.name})` : "")
              )}
            </h3>
            <p className={styles.optionDescription}>Import Excel file</p>
          </button>
        </div>

        {/* Textarea */}
        <textarea
          rows="5"
          cols="60"
          placeholder="Type your user story here... (use | to separate multiple)"
          value={userStoriesInput}
          onChange={(e) => {
            setUserStoriesInput(e.target.value);
            setSelectedFile(null);
          }}
          className={styles.textArea}
        ></textarea>

        {error && <p className={styles.errorText}>{error}</p>}

        <div className={styles.testTypesContainer}>
          <h4 className={styles.testTypesTitle}>Select Test Types to Generate</h4>
          <p className={styles.testTypesHint}>
            No selection is prefilled. Choose only the test suites you currently need.
          </p>
          <div className={styles.checkboxGroup}>
            {TEST_TYPE_OPTIONS.map((option) => (
              <label key={option.value} className={styles.checkboxLabel}>
                <input
                  type="checkbox"
                  value={option.value}
                  checked={selectedTestTypes.includes(option.value)}
                  onChange={handleTestTypeChange}
                />
                {option.label}
              </label>
            ))}
          </div>
        </div>

        {/* Generate */}
        <div className={styles.generateButtonContainer}>
          <button onClick={fetchTestCases} className={styles.generateButton}>
            {loadingGeneration ? <div className={styles.spinner}></div> : "Generate Test Cases"}
          </button>
        </div>

        {/* Results */}
        {Array.isArray(testCases) && testCases.length > 0 && (
          <>
            <div className={styles.gitPushContainer}>
              <button
                onClick={handlePushToGitClick}
                disabled={isPushingToGit}
                className={styles.gitPushButton}
              >
                {isPushingToGit ? <div className={styles.spinner}></div> : "Push to Git"}
              </button>
            </div>

            {testCases.map((tc, idx) => (
              <div key={idx} className={styles.testCaseCard}>
                <div className={styles.testCaseHeader}>
                  <h4 className={styles.testCaseTitle}>Generated Test Case : {idx + 1}</h4>
                </div>
                <table className={styles.testCaseTable}>
                  <thead>
                    <tr>
                      <th>User Story</th>
                      <th>Automated Test Cases</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td className={`${styles.testCaseTableTd} ${styles.userStoryCell}`}>
                        {/* Backend returns Prompt with the story; fall back to textarea */}
                        {tc.Prompt || userStoriesInput || tc.manual_testcase || "—"}
                      </td>
                      <td className={`${styles.testCaseTableTd} ${styles.code}`}>
                        <pre>
                          <code>{tc.auto_testcase || "No output generated"}</code>
                        </pre>
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>
            ))}
          </>
        )}
      </div>

      {/* Git Modal */}
      {showGitModal && (
        <div className={styles.modalOverlay}>
          <div className={styles.modalContent}>
            <h2>Push to Git</h2>
            <button onClick={handleGitModalClose} className={styles.closeModalButton}>
              <i className="fa-solid fa-xmark"></i>
            </button>
            <label>
              Repository URL:
              <input
                type="text"
                value={repoUrl}
                onChange={(e) => setRepoUrl(e.target.value)}
                placeholder="e.g., https://github.com/user/repo.git"
              />
            </label>
            <label>
              Branch Name:
              <input
                type="text"
                value={branchName}
                onChange={(e) => setBranchName(e.target.value)}
                placeholder="e.g., main or feature/my-tests"
              />
            </label>
            <label>
              Commit Message:
              <textarea
                rows="3"
                value={commitMessage}
                onChange={(e) => setCommitMessage(e.target.value)}
                placeholder="Enter commit message"
              ></textarea>
            </label>
            <div className={styles.modalActions}>
              <button onClick={handleGitModalClose} className={styles.modalCancelButton} disabled={isPushingToGit}>
                Cancel
              </button>
              <button onClick={handleGitModalSubmit} className={styles.modalSubmitButton} disabled={isPushingToGit}>
                {isPushingToGit ? <div className={styles.spinner}></div> : "Push"}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Nav */}
      <div className={styles.navigationButtons}>
        <button onClick={onBack} className={styles.navButton}>
          <i className="fa-solid fa-angle-left"></i>
          Previous
        </button>

        <button onClick={onNext} className={`${styles.navButton} ${styles.next}`}>
          Next <i className="fa-solid fa-angle-right"></i>
        </button>
      </div>
    </div>
  );
};

export default StoryInput;
