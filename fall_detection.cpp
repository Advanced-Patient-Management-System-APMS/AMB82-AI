#include "WiFi.h"
#include "StreamIO.h"
#include "VideoStream.h"
#include "RTSP.h"
#include "NNObjectDetection.h"
#include "VideoStreamOverlay.h"
#include "ObjectClassList.h"
#include <algorithm>   // std::max, std::min 사용

#define CHANNEL     0    // OSD(박스/텍스트) 포함 RTSP용 채널
#define CHANNEL_RAW 1    // OSD 없는 깨끗한 RTSP용 채널
#define CHANNELNN   3    // NN 입력용 RGB 채널

// 학습 입력과 맞추는 해상도(예: 416으로 학습했다면 416 권장)
#define NNWIDTH    640
#define NNHEIGHT   640

// Wi-Fi
char ssid[] = "";
char pass[] = "";
int  status = WL_IDLE_STATUS;

// -------- 낙상 룰 파라미터 --------
static const float BED_IOU_TH   = 0.10f;  // BED와의 IoU가 이보다 작으면 '침대 밖'
static const float FALL_HOLD    = 1.5f;   // fall 지속시간(초)
static const int   SCORE_TH     = 40;     // fall score 임계값(0~100 가정)

// ObjectClassList.h의 라벨 인덱스에 맞춰 설정!
#define FALL_ID  0

// -------- 전역 --------
// CH(0) : OSD 포함 H.264
VideoSetting config(VIDEO_FHD, 30, VIDEO_H264, 0);
// CH_RAW(1) : OSD 없는 H.264 (해상도/프레임 동일하게)
VideoSetting configRaw(VIDEO_FHD, 30, VIDEO_H264, 0);
// CHNN(3) : NN 입력용 RGB
VideoSetting configNN(NNWIDTH, NNHEIGHT, 10, VIDEO_RGB, 0);

NNObjectDetection ObjDet;

// RTSP 서버 2개
RTSP rtsp;       // OSD 포함 영상용
RTSP rtsp_raw;   // OSD 없는 영상용

// StreamIO 파이프라인
StreamIO videoStreamer(1, 1);      // CH -> rtsp
StreamIO videoStreamerRaw(1, 1);   // CH_RAW -> rtsp_raw
StreamIO videoStreamerNN(1, 1);    // CHNN -> ObjDet

IPAddress ip;
int rtsp_portnum = 0;       // OSD 포함 RTSP 포트
int rtsp_raw_portnum = 0;   // OSD 없는 RTSP 포트

// BED ROI (RTSP 해상도 기준 픽셀)
static bool bed_roi_inited = false;
static int bed_x = 0, bed_y = 0, bed_w = 0, bed_h = 0;

// 지속 시간 타이머
static float fall_timer = 0.0f;

// ---- 유틸: xywh IoU ----
static float iou_xywh(int ax, int ay, int aw, int ah,
                      int bx, int by, int bw, int bh) {
  int ax2 = ax + aw, ay2 = ay + ah;
  int bx2 = bx + bw, by2 = by + bh;
  int inter_x1 = std::max(ax, bx), inter_y1 = std::max(ay, by);
  int inter_x2 = std::min(ax2, bx2), inter_y2 = std::min(ay2, by2);
  int iw = std::max(0, inter_x2 - inter_x1);
  int ih = std::max(0, inter_y2 - inter_y1);
  int inter = iw * ih;
  int A = aw * ah;
  int B = bw * bh;
  int uni = A + B - inter;
  return (uni > 0) ? (float)inter / (float)uni : 0.0f;
}

void setup() {
  Serial.begin(115200);

  // --- Wi-Fi ---
  while (status != WL_CONNECTED) {
    Serial.print("Attempting to connect to WPA SSID: ");
    Serial.println(ssid);
    status = WiFi.begin(ssid, pass);
    delay(2000);
  }
  ip = WiFi.localIP();

  // --- Camera / RTSP ---
  config.setBitrate(2 * 1024 * 1024);     // 2 Mbps (OSD 포함 스트림)
  configRaw.setBitrate(2 * 1024 * 1024);  // 2 Mbps (OSD 없는 스트림)

  // 채널 설정
  Camera.configVideoChannel(CHANNEL,     config);    // CH 0 : OSD 포함용 H.264
  Camera.configVideoChannel(CHANNEL_RAW, configRaw); // CH 1 : OSD 없는 H.264
  Camera.configVideoChannel(CHANNELNN,   configNN);  // CH 3 : NN 입력용 RGB
  Camera.videoInit();

  // --- RTSP: OSD 포함 스트림용 (CH 0) ---
  rtsp.configVideo(config);
  rtsp.begin();
  rtsp_portnum = rtsp.getPort();

  // --- RTSP: OSD 없는 스트림용 (CH_RAW 1) ---
  rtsp_raw.configVideo(configRaw);
  rtsp_raw.begin();
  rtsp_raw_portnum = rtsp_raw.getPort();

  // --- Object Detection ---
  ObjDet.configVideo(configNN);
  ObjDet.modelSelect(OBJECT_DETECTION, CUSTOMIZED_YOLOV7TINY, NA_MODEL, NA_MODEL);
  ObjDet.begin();

  // --- StreamIO: CH -> rtsp (OSD 포함 스트림) ---
  videoStreamer.registerInput(Camera.getStream(CHANNEL));
  videoStreamer.registerOutput(rtsp);
  if (videoStreamer.begin() != 0) {
    Serial.println("StreamIO link start failed (CHANNEL -> rtsp)");
  }
  Camera.channelBegin(CHANNEL);

  // --- StreamIO: CH_RAW -> rtsp_raw (OSD 없는 스트림) ---
  videoStreamerRaw.registerInput(Camera.getStream(CHANNEL_RAW));
  videoStreamerRaw.registerOutput(rtsp_raw);
  if (videoStreamerRaw.begin() != 0) {
    Serial.println("StreamIO link start failed (CHANNEL_RAW -> rtsp_raw)");
  }
  Camera.channelBegin(CHANNEL_RAW);

  // --- StreamIO: CHNN -> ObjDet ---
  videoStreamerNN.registerInput(Camera.getStream(CHANNELNN));
  videoStreamerNN.setStackSize();
  videoStreamerNN.setTaskPriority();
  videoStreamerNN.registerOutput(ObjDet);
  if (videoStreamerNN.begin() != 0) {
    Serial.println("StreamIO link start failed (CHANNELNN -> ObjDet)");
  }
  Camera.channelBegin(CHANNELNN);

  // --- OSD (CH 0에만 적용) ---
  OSD.configVideo(CHANNEL, config);
  OSD.begin();

  // --- 세로형 중앙 BED ROI 초기화 ---
  uint16_t im_h = config.height();
  uint16_t im_w = config.width();

  // 화면 중앙에 세로(portrait) 직사각형: 너비는 화면의 35%, 높이는 80% (원하면 조정)
  bed_w = (int)(im_w * 0.35f);
  bed_h = (int)(im_h * 0.80f);
  bed_x = (im_w - bed_w) / 2;
  bed_y = (im_h - bed_h) / 2;
  bed_roi_inited = true;

  Serial.println();
  Serial.print("RTSP URL (with OSD): rtsp://");
  Serial.print(ip);
  Serial.print(":");
  Serial.println(rtsp_portnum);

  Serial.print("RTSP URL (no OSD):  rtsp://");
  Serial.print(ip);
  Serial.print(":");
  Serial.println(rtsp_raw_portnum);
}

void loop() {
  // 메인 루프에서 즉시 결과 수신
  std::vector<ObjectDetectionResult> results = ObjDet.getResult();

  uint16_t im_h = config.height();
  uint16_t im_w = config.width();

  // (원하면 출력 주기 조정 — 기존 코드 유지)
  /*
  Serial.print("Network URL for RTSP Streaming (with OSD): rtsp://");
  Serial.print(ip);
  Serial.print(":");
  Serial.println(rtsp_portnum);
  Serial.print("Network URL for RTSP Streaming (no OSD):  rtsp://");
  Serial.print(ip);
  Serial.print(":");
  Serial.println(rtsp_raw_portnum);
  Serial.println(" ");
  */
  

  // OSD는 CH 0에만 생성/갱신
  OSD.createBitmap(CHANNEL);

  // --- BED ROI 시각화(세로형 중앙) ---
  if (bed_roi_inited) {
    OSD.drawRect(CHANNEL, bed_x, bed_y, bed_x + bed_w, bed_y + bed_h, 3, OSD_COLOR_YELLOW);
    OSD.drawText(CHANNEL, bed_x, bed_y - OSD.getTextHeight(CHANNEL),
                 (char*)"BED ROI center portrait", OSD_COLOR_YELLOW);
  }

  bool fall_cond_met = false; // 이번 프레임에서 '침대 밖 + fall' 조건 충족?

  int n = ObjDet.getResultCount();
  //printf("Total number of objects detected = %d\r\n", n);

  for (int i = 0; i < n; i++) {
    int obj_type = results[i].type();

    // ObjectClassList.h에서 filter=1인 클래스만 사용
    if (!itemList[obj_type].filter) continue;

    ObjectDetectionResult item = results[i];

    // 0~1.0 좌표 → 픽셀 좌표
    int xmin = (int)(item.xMin() * im_w);
    int xmax = (int)(item.xMax() * im_w);
    int ymin = (int)(item.yMin() * im_h);
    int ymax = (int)(item.yMax() * im_h);

    // 박스 그리기/텍스트 (CH 0에만)
    OSD.drawRect(CHANNEL, xmin, ymin, xmax, ymax, 3, OSD_COLOR_WHITE);
    char text_str[32];
    snprintf(text_str, sizeof(text_str), "%s %d",
             itemList[obj_type].objectName, item.score());
    OSD.drawText(CHANNEL, xmin,
                 std::max(0, ymin - OSD.getTextHeight(CHANNEL)),
                 text_str, OSD_COLOR_CYAN);

    // --- 낙상 판단: fall 클래스만 ---
    if (obj_type == FALL_ID && item.score() >= SCORE_TH) {
      int pw = std::max(1, xmax - xmin);
      int ph = std::max(1, ymax - ymin);

      float bed_iou = 0.0f;
      if (bed_roi_inited) {
        bed_iou = iou_xywh(xmin, ymin, pw, ph, bed_x, bed_y, bed_w, bed_h);
      }

      // 침대 밖?
      bool cond_out_bed = (bed_iou < BED_IOU_TH);

      // 디버그 정보 (CH 0에만)
      char info[48];
      snprintf(info, sizeof(info), "BED IoU:%.2f", bed_iou);
      OSD.drawText(CHANNEL, xmin,
                   std::min(im_h - OSD.getTextHeight(CHANNEL), ymax + 4),
                   info, OSD_COLOR_WHITE);

      if (cond_out_bed) {
        fall_cond_met = true;
      }
    }
  }

  // --- 지속시간 타이머 (약 10fps 가정) ---
  const float dt = 0.1f;
  if (fall_cond_met) {
    fall_timer += dt;
  } else {
    fall_timer = 0.0f;
  }

  // --- 경고 ---
  if (fall_timer >= FALL_HOLD) {
    OSD.drawText(CHANNEL, 20, 40, (char*)"FALL SUSPECT", OSD_COLOR_RED);
    Serial.println("FALL_SUSPECTED");
  }

  // OSD는 CH 0에만 업데이트
  OSD.update(CHANNEL);

  // 새 결과 대기
  delay(100);
}
