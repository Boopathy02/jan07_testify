import React, { useRef } from "react";
import styles from "../css/ImageHandles.module.css";

const ImageDragDrop = ({ files, setFiles }) => {
  const dragItem = useRef();
  const dragOverItem = useRef();

  const handleDragStart = (index) => (dragItem.current = index);
  const handleDragEnter = (index) => (dragOverItem.current = index);
  const handleDragEnd = () => {
    const newList = [...files];
    const dragged = newList[dragItem.current];
    newList.splice(dragItem.current, 1);
    newList.splice(dragOverItem.current, 0, dragged);
    dragItem.current = null;
    dragOverItem.current = null;
    setFiles(newList);
  };

  const removeImage = (index) => {
    const newFiles = [...files];
    if (newFiles[index]?.preview) {
      URL.revokeObjectURL(newFiles[index].preview);
    }
    newFiles.splice(index, 1);
    setFiles(newFiles);
  };

  return (
    <div className={styles.imageDragDropContainer}>
      {files.map((file, idx) => (
        <div
          key={file.name + idx}
          draggable
          onDragStart={() => handleDragStart(idx)}
          onDragEnter={() => handleDragEnter(idx)}
          onDragEnd={handleDragEnd}
          onDragOver={(e) => e.preventDefault()}
          className={styles.imageCard}
        >
          <img src={file.preview} alt={file.name} className={styles.imagePreview} />
          <div className={styles.imageName}>{file.name}</div>

          {/* Index */}
          <div className={styles.imageIndex}>{idx + 1}</div>

          {/* Remove */}
          <button onClick={() => removeImage(idx)} className={styles.removeButton}>×</button>
        </div>
      ))}
    </div>
  );
};

export default ImageDragDrop;
