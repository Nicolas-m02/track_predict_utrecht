#%%
import asyncio
import base64
import numpy as np
import cv2
from app import broadcast

async def generate_stream():
    t = 0
    while True:
        # Simulated frame
        img = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.putText(img, f"t={t}", (50,120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        _, buffer = cv2.imencode('.jpg', img)
        frame_b64 = base64.b64encode(buffer).decode()

        # Simulated time-series
        value = float(np.sin(t * 0.1))

        await broadcast({
            "frame": frame_b64,
            "value": value,
            "t": t
        })

        t += 1
        await asyncio.sleep(0.01) 

        print(t)

if __name__ == "__main__":
    asyncio.run(generate_stream())