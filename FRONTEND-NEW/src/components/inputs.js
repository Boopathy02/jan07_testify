import { useState, useEffect } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import ImageUpload from "./imageuploads";
import StoryInput from "./storyinput";
import URLInput from "./urlinput";
import Execute from "./execute"; // 👈 Add your final step component here
import styles from "../css/Inputs.module.css";

const Input = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const projectName = location.state?.projectName;
  const projectId = location.state?.projectId;
  const stateFlow = location.state?.flow;
  const storedFlow = sessionStorage.getItem("inputFlow");
  const storedStartFlow = sessionStorage.getItem("inputStartFlow");
  const flowFromPath = location.pathname.endsWith("/input/url") ? "url" : null;
  const flowType = stateFlow || flowFromPath || storedStartFlow || storedFlow;
  const isUrlFlow = flowType === "url";
  
  
  const [currentStep, setCurrentStep] = useState(1);
  const [persistedFiles, setPersistedFiles] = useState([]);
  const [pageNames, setPageNames] = useState([]);
  const [testCases, setTestCases] = useState([]);

  useEffect(() => {
    if (projectName) {
      localStorage.setItem("activeProjectName", projectName);
    }
    if (projectId) {
      localStorage.setItem("projectId", String(projectId));
    }
  }, [projectName, projectId]);

  useEffect(() => {
    if (flowType) {
      sessionStorage.setItem("inputFlow", flowType);
    }
  }, [flowType]);

  const stepFromPath = (path) => {
    if (isUrlFlow) {
      if (path.endsWith("/input/url")) return 1;
      if (path.endsWith("/input/upload")) return 2;
      if (path.endsWith("/input/story")) return 3;
      if (path.endsWith("/input/execute")) return 4;
      return 1;
    }
    if (path.endsWith("/input/upload")) return 1;
    if (path.endsWith("/input/story")) return 2;
    if (path.endsWith("/input/url")) return 3;
    if (path.endsWith("/input/execute")) return 4;
    return 1;
  };

  const pathFromStep = (step) => {
    if (isUrlFlow) {
      switch (step) {
        case 1:
          return "/input/url";
        case 2:
          return "/input/upload";
        case 3:
          return "/input/story";
        case 4:
          return "/input/execute";
        default:
          return "/input/url";
      }
    }
    switch (step) {
      case 1:
        return "/input/upload";
      case 2:
        return "/input/story";
      case 3:
        return "/input/url";
      case 4:
        return "/input/execute";
      default:
        return "/input/upload";
    }
  };

  useEffect(() => {
    const nextStep = stepFromPath(location.pathname || "");
    if (nextStep !== currentStep) {
      setCurrentStep(nextStep);
    }
  }, [location.pathname, currentStep]);

  const handleNext = () => {
    const nextStep = Math.min(4, currentStep + 1);
    const nextState = flowType
      ? { ...(location.state || {}), flow: flowType }
      : location.state;
    navigate(pathFromStep(nextStep), { state: nextState });
    setCurrentStep(nextStep);
  };

  const handleBack = () => {
    const prevStep = Math.max(1, currentStep - 1);
    const nextState = flowType
      ? { ...(location.state || {}), flow: flowType }
      : location.state;
    navigate(pathFromStep(prevStep), { state: nextState });
    setCurrentStep(prevStep);
  };

  const renderStep = () => {
    if (isUrlFlow) {
      switch (currentStep) {
        case 1:
          return <URLInput onBack={handleBack} onNext={handleNext} apiMode="url" />;
        case 2:
          return (
            <ImageUpload
              handleNext={handleNext}
              persistedFiles={persistedFiles}
              setPersistedFiles={setPersistedFiles}
              pageNames={pageNames}
              setPageNames={setPageNames}
              projectName={projectName}
              projectId={projectId}
            />
          );
        case 3:
          return (
            <StoryInput
              onBack={handleBack}
              onNext={handleNext}
              testCases={testCases}
              setTestCases={setTestCases}
              projectName={projectName}
            />
          );
        case 4:
          return <Execute onBack={handleBack} />;
        default:
          return null;
      }
    }

    switch (currentStep) {
      case 1:
        return (
          <ImageUpload
            handleNext={handleNext}
            persistedFiles={persistedFiles}
            setPersistedFiles={setPersistedFiles}
            pageNames={pageNames}
            setPageNames={setPageNames}
            projectName={projectName}
            projectId={projectId}
          />
        );
      case 2:
        return (
          <StoryInput
            onBack={handleBack}
            onNext={handleNext}
            testCases={testCases}
            setTestCases={setTestCases}
            projectName={projectName}
          />
        );
      case 3:
        return <URLInput onBack={handleBack} onNext={handleNext} apiMode="ocr" />;
      case 4:
        return <Execute onBack={handleBack} />;
      default:
        return null;
    }
  };

  return (
    <div>
      <div className={styles.inputContainer}>
        <nav className={styles.navbar}>
          <div className={styles.navbarBrand}>
            <i className={`fa fa-code ${styles.navbarIcon}`}></i>
            <div>
              <h4 className={styles.navbarTitle}>AutoTest Studio</h4>
              <p className={styles.navbarSubtitle}>
                Automation Development Platform
              </p>
            </div>
          </div>

          <button
            onClick={() => navigate("/")}
            className={styles.backButton}
          >
            Back to Dashboard
          </button>
        </nav>

        <h1 className={styles.wizardTitle}>
          Project Setup Wizard
        </h1>

        <div className={styles.stepContainer}>
          <div className={styles.stepItem}>
            <div className={styles.stepIconContainer}>
              <i className={`fa-solid fa-arrow-up-from-bracket ${styles.stepIcon}`}></i>
            </div>
            <div className={styles.stepTextContainer}>
              <h2>Upload Design</h2>
              <p>
                Upload screenshots or visual designs of your application
              </p>
            </div>
          </div>

          <div className={styles.stepItem}>
            <div className={styles.stepIconContainer}>
              <i className={`fa-regular fa-message ${styles.stepIcon}`}></i>
            </div>
            <div className={styles.stepTextContainer}>
              <h2>Import User Stories</h2>
              <p>
                Add user stories from Jira, Excel, or create them manually
              </p>
            </div>
          </div>

          <div className={styles.stepItem}>
            <div className={styles.stepIconContainer}>
              <i className={`fa-solid fa-code ${styles.stepIcon}`}></i>
            </div>
            <div className={styles.stepTextContainer}>
              <h2>Generate Scripts</h2>
              <p>
                Configure framework and generate test scripts
              </p>
            </div>
          </div>

          <div
            className={styles.stepItem}
            onClick={() => navigate('/prompts', { state: { projectName, projectId } })}
          >
            <div className={styles.stepIconContainer}>
              <i className={`fa-solid fa-file-pen ${styles.stepIcon}`}></i>
            </div>
            <div className={styles.stepTextContainer}>
              <h2>Edit Prompts</h2>
              <p>
                Customize the AI prompts for the current project
              </p>
            </div>
          </div>
        </div>

        {renderStep()}
      </div>
    </div>
  );
};

export default Input;
