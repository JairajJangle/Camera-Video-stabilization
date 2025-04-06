import cv2
import numpy as np

CAM = 0

class Tracker:
    def __init__(self):
        self.tracked_features = []
        self.prev_gray = None
        self.fresh_start = True
        # 3x3 identity matrix for rigid transform (affine 2x3 in a 3x3 matrix)
        self.rigid_transform = np.eye(3, dtype=np.float32)

    def process_image(self, img):
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # If we need more features
        if len(self.tracked_features) < 200:
            # goodFeaturesToTrack returns array of shape (N, 1, 2)
            corners = cv2.goodFeaturesToTrack(gray, maxCorners=300, qualityLevel=0.01, minDistance=1)
            if corners is not None:
                print(f"found {len(corners)} features")
                # Convert corners to list of (x,y) points
                corners = corners.reshape(-1, 2)
                self.tracked_features.extend(corners.tolist())

        if self.prev_gray is not None and len(self.tracked_features) > 0:
            # Convert tracked_features to numpy array for optical flow
            prev_pts = np.array(self.tracked_features, dtype=np.float32).reshape(-1, 1, 2)
            
            # Calculate optical flow
            corners, status, errors = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, gray, prev_pts, None, winSize=(10, 10)
            )
            
            # Count successful tracks
            successful_tracks = np.count_nonzero(status)
            
            if successful_tracks < len(status) * 0.8:
                print("cataclysmic error")
                self.rigid_transform = np.eye(3, dtype=np.float32)
                self.tracked_features.clear()
                self.prev_gray = None
                self.fresh_start = True
                return
            else:
                self.fresh_start = False

            # Estimate rigid transform
            good_prev = prev_pts[status == 1].reshape(-1, 2)
            good_new = corners[status == 1].reshape(-1, 2)
            
            new_rigid_transform = cv2.estimateAffine2D(good_prev, good_new)[0]
            if new_rigid_transform is not None:
                # Convert 2x3 transform to 3x3
                nrt33 = np.eye(3, dtype=np.float32)
                nrt33[0:2, :] = new_rigid_transform
                self.rigid_transform = self.rigid_transform @ nrt33

            # Update tracked features
            self.tracked_features = good_new.tolist()

        # Draw tracked features
        for point in self.tracked_features:
            x, y = map(int, point)
            cv2.circle(img, (x, y), 3, (0, 0, 255), -1)  # -1 for filled circle

        # Update previous frame
        self.prev_gray = gray.copy()

def main():
    vc = cv2.VideoCapture(CAM)
    tracker = Tracker()
    
    print("in main")

    while True:
        ret, frame = vc.read()
        if not ret:
            break
            
        orig = frame.copy()
        tracker.process_image(orig)

        # Calculate inverse transform and warp
        inv_trans = cv2.invertAffineTransform(tracker.rigid_transform[0:2, :])
        orig_warped = cv2.warpAffine(orig, inv_trans, (frame.shape[1], frame.shape[0]))

        cv2.imshow("orig", orig_warped)
        if cv2.waitKey(1) & 0xFF == 27:  # Exit on ESC
            break

    vc.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()