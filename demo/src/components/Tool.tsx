import React, { useContext, useEffect, useState } from "react";
import AppContext from "./hooks/createContext";
import { ToolProps } from "./helpers/Interfaces";
import * as _ from "underscore";

interface ToolProps {
  handleMouseMove: (e: any) => void;
  isVideo: boolean;
}

const Tool = ({ handleMouseMove, isVideo }: ToolProps) => {
  const {
    image: [image],
    maskImg: [maskImg, setMaskImg],
  } = useContext(AppContext)!;

  const [shouldFitToWidth, setShouldFitToWidth] = useState(true);
  const bodyEl = document.body;
  const fitToPage = () => {
    if (!image) return;
    const imageAspectRatio = image.width / image.height;
    const screenAspectRatio = window.innerWidth / window.innerHeight;
    setShouldFitToWidth(imageAspectRatio > screenAspectRatio);
  };
  const resizeObserver = new ResizeObserver((entries) => {
    for (const entry of entries) {
      if (entry.target === bodyEl) {
        fitToPage();
      }
    }
  });
  useEffect(() => {
    fitToPage();
    resizeObserver.observe(bodyEl);
    return () => {
      resizeObserver.unobserve(bodyEl);
    };
  }, [image]);

  const imageClasses = "";
  const maskImageClasses = `absolute opacity-40 pointer-events-none`;

  return (
    <>
      {isVideo ? (
        <video
          onMouseMove={handleMouseMove}
          onMouseOut={() => _.defer(() => setMaskImg(null))}
          onTouchStart={handleMouseMove}
          src={image?.src}
          className={`${shouldFitToWidth ? "w-full" : "h-full"} ${imageClasses}`}
          autoPlay
          loop
          muted
        ></video>
      ) : (
        <>
          {image && (
            <img
              onMouseMove={handleMouseMove}
              onMouseOut={() => _.defer(() => setMaskImg(null))}
              onTouchStart={handleMouseMove}
              src={image.src}
              className={`${shouldFitToWidth ? "w-full" : "h-full"} ${imageClasses}`}
            ></img>
          )}
          {maskImg && (
            <img
              src={maskImg.src}
              className={`${shouldFitToWidth ? "w-full" : "h-full"} ${maskImageClasses}`}
            ></img>
          )}
        </>
      )}
    </>
  );
};

export default Tool;
