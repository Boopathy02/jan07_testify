import React, { useEffect, useState, useCallback } from "react";
import axios from "axios";
import API_BASE_URL from "../config";
import { toast } from "react-toastify";
import ImageDragDrop from "./imagehandles"; // adjust path if needed
import styles from "../css/ImageUpload.module.css";

const ImageUpload = ({
  handleNext,
  persistedFiles,
  setPersistedFiles,
  pageNames,
  setPageNames,
  projectName,
  projectId,
}) => {
  const [loadingIngestion, setLoadingIngestion] = useState(false);
  const [loadingMethods, setLoadingMethods] = useState(false);
  const [ingestionSuccess, setIngestionSuccess] = useState(false);
  const [error, setError] = useState("");
  const [activeProjectId, setActiveProjectId] = useState(() => projectId || localStorage.getItem("projectId"));

  const selectedFiles = persistedFiles;
  const setSelectedFiles = setPersistedFiles;

  useEffect(() => {
    if (projectId) {
      const id = String(projectId);
      setActiveProjectId(id);
      localStorage.setItem("projectId", id);
    }
    if (projectName) {
      localStorage.setItem("activeProjectName", projectName);
    }
  }, [projectId, projectName]);

  const ensureActiveProject = useCallback(async () => {
    if (activeProjectId) {
      return activeProjectId;
    }
    const token = localStorage.getItem("token");
    const name = projectName || localStorage.getItem("activeProjectName");
    if (!token || !name) {
      return null;
    }
    try {
      const response = await fetch(`${API_BASE_URL}/projects/activate`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({ project_name: name }),
      });
      if (!response.ok) {
        const txt = await response.text().catch(() => null);
        throw new Error(txt || `Failed to activate project (${response.status})`);
      }
      const payload = await response.json();
      const newId = payload?.project?.id ? String(payload.project.id) : null;
      if (newId) {
        setActiveProjectId(newId);
        localStorage.setItem("projectId", newId);
      }
      return newId;
    } catch (err) {
      console.error("Failed to activate project before upload:", err);
      toast.error(err.message || "Failed to activate project.");
      return null;
    }
  }, [activeProjectId, projectName]);

  const handleFileChange = async (e) => {
    const inputFiles = Array.from(e.target.files);
    let allProcessedFiles = [];

    for (const file of inputFiles) {
      if (file.type.startsWith("image/")) {
        const previewFile = new File([file], file.name, { type: file.type });
        previewFile.preview = URL.createObjectURL(previewFile);
        allProcessedFiles.push(previewFile);
      } else if (file.name.endsWith(".zip")) {
        try {
          const JSZip = (await import("jszip")).default;
          const zip = await JSZip.loadAsync(file);

          for (const zipEntry of Object.values(zip.files)) {
            if (
              !zipEntry.dir &&
              /\.(jpe?g|png|gif|bmp|webp)$/i.test(zipEntry.name)
            ) {
              const blob = await zipEntry.async("blob");
              const imageFile = new File([blob], zipEntry.name, {
                type: blob.type,
              });
              imageFile.preview = URL.createObjectURL(imageFile);
              allProcessedFiles.push(imageFile);
            }
          }
        } catch (err) {
          toast.error("Failed to extract ZIP file.");
        }
      } else {
        toast.warn(`${file.name} is not a valid image or ZIP file.`);
      }
    }

    setSelectedFiles((prev) => [...prev, ...allProcessedFiles]);
  };

  const handleContinue = async () => {
    setLoadingIngestion(true);
    setError("");
    setIngestionSuccess(false);

    if (selectedFiles.length === 0) {
      toast.warn("Please upload at least one file.");
      setLoadingIngestion(false);
      return;
    }

    const formData = new FormData();
    selectedFiles.forEach((file) => {
      formData.append("images", file);
    });

    const orderedImageNames = selectedFiles.map((file) => file.name);
    formData.append("ordered_images", JSON.stringify({ ordered_images: orderedImageNames }));

    try {
      const activeId = await ensureActiveProject();
      if (!activeId) {
        throw new Error("No active project. Please start a project first.");
      }
      const response = await axios.post(
        `${API_BASE_URL}/upload-image`,
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        }
      );

      if (response.status === 200) {
        toast.success("OCR extracted and stored in ChromaDB successfully.");
        setIngestionSuccess(true);
      }
    } catch (error) {
      console.error("Error uploading files:", error);
      setError(error?.message || "Please try again.");
      toast.error(`Error uploading files: ${error?.message || "Please try again."}`);
    } finally {
      setLoadingIngestion(false);
    }
  };

  const handleGenerateMethods = async () => {
    setLoadingMethods(true);
    try {
      const response = await fetch(`${API_BASE_URL}/rag/generate-page-methods`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({}),
      });

      const data = await response.json();
      const names = Object.keys(data);
      setPageNames(names);
      toast.success("Successfully generated methods");
    } catch (error) {
      console.error("Error fetching page methods:", error);
      setPageNames([]);
      toast.error("Error generating methods");
    } finally {
      setLoadingMethods(false);
    }
  };

  return (
    <div className={styles.imageUploadContainer}>
      <div className={styles.uploadBox}>
        <h2 className={styles.uploadTitle}>Upload Designs</h2>
        <p className={styles.uploadSubtitle}>
          Upload screenshots or visual designs of your application
        </p>

        <div className={styles.dropzone}>
          <input
            id="file-upload"
            type="file"
            accept="image/*,.zip"
            multiple
            style={{ display: "none" }}
            onChange={handleFileChange}
          />

          <i className={`fa-solid fa-cloud-arrow-up ${styles.uploadIcon}`}></i>

          <h3 className={styles.uploadText}>
            Upload Design Files
          </h3>

          <p className={styles.uploadInstructions}>
            Click the button below to select your files
          </p>

          <button
            onClick={() => document.getElementById("file-upload").click()}
            className={styles.selectFilesButton}
          >
            <i className={`fa-solid fa-upload ${styles.selectFilesButtonIcon}`}></i>
            <span className={styles.selectFilesButtonText}>Select Files</span>
          </button>
        </div>

        {selectedFiles.length > 0 && (
          <div style={{ marginTop: "20px" }}>
            <ImageDragDrop files={selectedFiles} setFiles={setSelectedFiles} />
          </div>
        )}

        {error && <p className={styles.errorText}>{error}</p>}

        {ingestionSuccess && (
          <div className={styles.successMessage}>
            ✅ OCR extracted and stored in ChromaDB successfully.
          </div>
        )}

        {pageNames.length > 0 && (
          <div className={styles.availablePagesContainer}>
            <h5 className={styles.availablePagesTitle}>Available Pages:</h5>
            <ul className={styles.pageList}>
              {pageNames.map((name, idx) => (
                <li
                  key={idx}
                  className={styles.pageListItem}
                >
                  {name}
                </li>
              ))}
            </ul>
          </div>
        )}

        <div className={styles.actionButtons}>
          <button
            onClick={handleContinue}
            disabled={loadingIngestion}
            className={styles.uploadImagesButton}
          >
            {loadingIngestion ? <div className={styles.spinner}></div> : "Upload Images"}
          </button>

          <button
            onClick={handleGenerateMethods}
            disabled={loadingMethods}
            className={styles.generateMethodsButton}
          >
            {loadingMethods ? <div className={styles.spinner}></div> : "Generate Page Methods"}
          </button>
        </div>
      </div>

      <div className={styles.nextButtonContainer}>
        <button
          onClick={handleNext}
          className={styles.nextButton}
        >
          Next <i className="fa-solid fa-angle-right"></i>
        </button>
      </div>
    </div>
  );
};

export default ImageUpload;

