import cv2

cap = cv2.VideoCapture('GP066349.MP4')
ret, current_frame = cap.read()
previous_frame = current_frame
frames = 0

while(cap.isOpened()):
    if current_frame is not None and previous_frame is not None:
        current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        previous_frame_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)    
    
        frame_diff = cv2.absdiff(current_frame_gray,previous_frame_gray)
        frame_diff = cv2.convertScaleAbs(frame_diff, alpha=3, beta=0)
        cv2.imshow('frame diff ',frame_diff)      
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    if current_frame is not None:
        previous_frame = current_frame.copy()
    else:
        previous_frame = None
    ret, current_frame = cap.read()
    frames += 1
    if frames > 1000:
        break

cap.release()
cv2.destroyAllWindows()

# import cv2

# cap = cv2.VideoCapture('GP066349.MP4')
# frames = 0
# while(cap.isOpened()):
#     # Capture frame-by-frame
#     ret, frame = cap.read()
#     print(ret, frame)
#     frames += 1
#     if ret:
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         cv2.imshow("Output", gray)
#         if cv2.waitKey(33) & 0xFF == ord('q'):
#             break
#     else:
#         continue
#     if frames > 1000:
#         break
# cap.release()
# cv2.destroyAllWindows()
# print(frames)