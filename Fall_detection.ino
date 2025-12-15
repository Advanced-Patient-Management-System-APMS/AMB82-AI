/*
  AMB82-mini Fall Detection System (Final Version)
  
  [기능 요약]
  1. AI 모델: YOLOv7-tiny를 사용하여 사람(Person)을 실시간 감지
  2. 듀얼 스트림: 
     - Channel 0 (OSD): 분석 결과(박스, 텍스트)가 그려진 화면
     - Channel 1 (Clean): 원본 깨끗한 화면 (녹화용)
  3. 낙상 감지 알고리즘 (순차적 로직):
     - 단계 1: 화면 위아래가 동시에 잘리지 않았는지 확인 (얼빡샷 필터)
     - 단계 2: 사람 키 대비 1.2배 이상의 속도로 하강했는지 감지 (충격 감지)
     - 단계 3: 하강 후 납작한 비율(누운 자세)이 되었는지 확인
     - 단계 4: 그 상태가 1초 이상 유지되면 최종 낙상 확정
*/

#include "WiFi.h"
#include "StreamIO.h"
#include "VideoStream.h"
#include "RTSP.h"
#include "NNObjectDetection.h"
#include "VideoStreamOverlay.h"
#include "ObjectClassList.h"

// ------------------------------------------------------------
// [채널 설정]
// ------------------------------------------------------------
// CHANNEL_OSD (0번): 박스와 글씨가 그려질 채널 (RTSP 포트 554)
// CHANNEL_CLEAN (1번): 아무것도 없는 원본 채널 (RTSP 포트 555)
// CHANNELNN (3번): AI가 분석할 데이터 전용 채널 (화면에 안 보임)
#define CHANNEL_OSD   0  
#define CHANNEL_CLEAN 1  
#define CHANNELNN     3

// AI 모델 입력 해상도 (YOLOv7-tiny 권장 해상도)
#define NNWIDTH  640
#define NNHEIGHT 640

// 비디오 설정 객체 (FHD 해상도, 30fps)
VideoSetting configOSD(VIDEO_FHD, 30, VIDEO_H264, 0);
VideoSetting configClean(VIDEO_FHD, 30, VIDEO_H264, 0); 
VideoSetting configNN(NNWIDTH, NNHEIGHT, 10, VIDEO_RGB, 0); // AI는 RGB 포맷 사용

// 객체 생성
NNObjectDetection ObjDet; // AI 감지 객체
RTSP rtspOSD;             // OSD 스트림용 RTSP
RTSP rtspClean;           // Clean 스트림용 RTSP

// 데이터 연결 파이프라인 (StreamIO)
StreamIO videoStreamerOSD(1, 1);   // Cam -> RTSP OSD
StreamIO videoStreamerClean(1, 1); // Cam -> RTSP Clean
StreamIO videoStreamerNN(1, 1);    // Cam -> AI Model

char ssid[] = "Your_SSID";    // 와이파이 이름 (여기에 입력)
char pass[] = "Your_PASS";    // 와이파이 비번 (여기에 입력)
int status = WL_IDLE_STATUS;

IPAddress ip;
int rtsp_port_osd;
int rtsp_port_clean;

// =============================================================
// [튜닝 파라미터] 낙상 감지의 민감도를 조절하는 핵심 변수들
// =============================================================

// 1. 상대 속도 임계값 (Relative Velocity Threshold)
// 의미: 1초 동안 "자기 키(Height)"의 1.2배 거리만큼 아래로 이동하면 "급격한 하강"으로 간주
// 값이 클수록 둔감해지고(진짜 쾅! 넘어질 때만 감지), 작으면 앉는 동작도 감지함.
static const float REL_VEL_TH      = 1.20f;   

// 2. 가장자리 여유분 (Pixel)
// 화면 끝에서 10픽셀 이내에 박스가 닿으면 "닿았다"고 판정
static const int   BORDER_MARGIN   = 10;      

// 3. 낙상 비율 (Aspect Ratio Threshold)
// 박스 높이(Height) / 너비(Width) 비율. 
// 0.90 이하면 납작한 직사각형(누운 상태)으로 간주.
static const float FALL_AR_TH      = 0.90f;   

// 4. 최소 박스 크기
// 화면 전체 면적의 2% 미만인 작은 사람(멀리 있는 사람)은 무시
static const float MIN_BOX_AREA_FRAC = 0.02f; 

// 5. 낙상 유지 시간 (Hold Time)
// 낙상 상태가 1.0초 이상 지속되어야 최종 알람 발생 (잠깐 튀는 오탐 방지)
static const float FALL_HOLD_SEC   = 1.0f;    

// 6. 최소 이동 거리 (Noise Filter)
// 프레임 간 10픽셀 이상 움직여야 속도 계산 (카메라 노이즈 무시)
static const int   MIN_DROP_DIST   = 10;      

// 7. 충격 유효 시간 (Impact Window)
// 급격한 하강 감지 후 2.5초 안에 눕지 않으면 하강 이벤트 취소 (다시 일어난 것으로 간주)
static const uint32_t IMPACT_WINDOW_MS = 2500; 

// =============================================================
// [상태 변수] 로직 흐름을 제어하는 변수들
// =============================================================
static bool  fall_confirmed = false;   // 최종 낙상 확정 여부 (Red Box 표시용)
static bool  fall_alert_sent = false;  // 알림 중복 전송 방지 플래그
static bool  was_rapid_drop = false;   // "최근에 급격한 하강이 있었는가?" (메모리)
static uint32_t rapid_drop_ts = 0;     // 하강 감지 시각
static uint32_t lying_start_ms = 0;    // 누워있는 상태 시작 시각
static int      prev_cy = -1;          // 이전 프레임의 중심점 Y좌표
static uint32_t prev_time_ms = 0;      // 이전 프레임의 시간
static float    current_rel_vel = 0.0f;// 현재 계산된 속도 (디버깅 표시용)

void setup()
{
    Serial.begin(115200);

    // 1. 와이파이 연결
    while (status != WL_CONNECTED) {
        status = WiFi.begin(ssid, pass);
        delay(2000);
    }
    ip = WiFi.localIP();

    // 2. 비트레이트 설정 (두 채널 동시 송출이므로 대역폭 관리 필요)
    configOSD.setBitrate(2 * 1024 * 1024);   // 2Mbps
    configClean.setBitrate(2 * 1024 * 1024); // 2Mbps

    // 3. 카메라 채널 초기화
    Camera.configVideoChannel(CHANNEL_OSD, configOSD);
    Camera.configVideoChannel(CHANNEL_CLEAN, configClean);
    Camera.configVideoChannel(CHANNELNN, configNN);
    Camera.videoInit();

    // 4. RTSP 서비스 시작
    rtspOSD.configVideo(configOSD);
    rtspOSD.begin();
    rtsp_port_osd = rtspOSD.getPort(); // 보통 554

    rtspClean.configVideo(configClean);
    rtspClean.begin();
    rtsp_port_clean = rtspClean.getPort(); // 보통 555

    // 5. AI 모델 로드 (YOLOv7-tiny)
    ObjDet.configVideo(configNN);
    ObjDet.modelSelect(OBJECT_DETECTION, DEFAULT_YOLOV7TINY, NA_MODEL, NA_MODEL);
    ObjDet.begin();

    // 6. 스트림 파이프라인 연결
    // (1) OSD 스트림: 카메라 -> RTSP OSD
    videoStreamerOSD.registerInput(Camera.getStream(CHANNEL_OSD));
    videoStreamerOSD.registerOutput(rtspOSD);
    videoStreamerOSD.begin();

    // (2) Clean 스트림: 카메라 -> RTSP Clean
    videoStreamerClean.registerInput(Camera.getStream(CHANNEL_CLEAN));
    videoStreamerClean.registerOutput(rtspClean);
    videoStreamerClean.begin();

    // (3) AI 스트림: 카메라 -> AI 엔진
    videoStreamerNN.registerInput(Camera.getStream(CHANNELNN));
    videoStreamerNN.setStackSize();
    videoStreamerNN.setTaskPriority();
    videoStreamerNN.registerOutput(ObjDet);
    videoStreamerNN.begin();

    // 7. 채널 및 OSD 시작
    Camera.channelBegin(CHANNEL_OSD);
    Camera.channelBegin(CHANNEL_CLEAN);
    Camera.channelBegin(CHANNELNN);

    OSD.configVideo(CHANNEL_OSD, configOSD);
    OSD.begin();
    
    // 8. RTSP 주소 시리얼 출력
    Serial.println("=== System Started (Final Version) ===");
    Serial.println("--------------------------------");
    Serial.print("1. OSD Stream (Box): rtsp://");
    Serial.print(ip);
    Serial.print(":");
    Serial.println(rtsp_port_osd);
    
    Serial.print("2. Clean Stream    : rtsp://");
    Serial.print(ip);
    Serial.print(":");
    Serial.println(rtsp_port_clean);
    Serial.println("--------------------------------");
}

void loop()
{
    // AI 결과 가져오기
    std::vector<ObjectDetectionResult> results = ObjDet.getResult();
    uint16_t im_h = configOSD.height();
    uint16_t im_w = configOSD.width();
    
    // OSD 비트맵 생성 (그리기 준비)
    OSD.createBitmap(CHANNEL_OSD);

    // ---------------------------------------------------------
    // STEP 1: 가장 큰 사람(Main Subject) 찾기
    // 여러 사람이 잡힐 경우, 카메라에 가장 크게 잡히는 사람을 대상으로 분석합니다.
    // ---------------------------------------------------------
    int bestPersonIdx = -1;
    float bestArea = 0.0f;

    for (int i = 0; i < ObjDet.getResultCount(); i++) {
        int obj_type = results[i].type();
        if (strcmp(itemList[obj_type].objectName, "person") != 0) continue;

        ObjectDetectionResult item = results[i];
        int w = (int)((item.xMax() - item.xMin()) * im_w);
        int h = (int)((item.yMax() - item.yMin()) * im_h);
        float area = (float)w * (float)h;

        if (area > bestArea) {
            bestArea = area;
            bestPersonIdx = i;
        }
    }

    // ---------------------------------------------------------
    // STEP 2: 낙상 감지 로직 수행
    // ---------------------------------------------------------
    bool is_lying_now = false;    // 현재 프레임에서 누워있는가?
    bool ignore_by_edge = false;  // 화면 잘림으로 인해 판단을 보류할 것인가?

    if (bestPersonIdx >= 0) {
        ObjectDetectionResult p = results[bestPersonIdx];

        // 좌표 변환 (0.0~1.0 -> 픽셀 좌표)
        int ymin = (int)(p.yMin() * im_h);
        int ymax = (int)(p.yMax() * im_h);
        int xmin = (int)(p.xMin() * im_w);
        int xmax = (int)(p.xMax() * im_w);
        
        int bw = xmax - xmin;      // 너비
        int bh = ymax - ymin;      // 높이
        int cy = ymin + (bh / 2);  // 중심점 Y

        // =============================================================
        // [로직 A] 엣지 체크 (화면 잘림 확인) - AND 조건 사용
        // =============================================================
        // 조건: 사람의 머리(Top)와 발(Bottom)이 동시에 화면 끝에 닿았는가?
        // 참(True): 카메라 바로 앞 얼빡샷 -> 키(Height) 계산 불가 -> 무시(Ignore)
        // 거짓(False): 바닥에 쓰러져서 발만 닿았거나, 공중에 떠 있거나 등 -> 정상 판단
        
        bool touch_top = (ymin <= BORDER_MARGIN);
        bool touch_bottom = (ymax >= (im_h - BORDER_MARGIN));

        if (touch_top && touch_bottom) {
            ignore_by_edge = true;
        }
        // =============================================================

        // [로직 B] 상대 속도(Relative Velocity) 계산
        uint32_t now = millis();
        float dt = (now - prev_time_ms) / 1000.0f;
        current_rel_vel = 0.0f;

        // 시간차가 유효할 때 계산
        if (prev_cy != -1 && dt > 0.0f && dt < 1.0f) {
            int dist_y = cy - prev_cy; 
            
            // 노이즈 필터: 10픽셀 이상 움직였을 때만 속도로 인정
            if (dist_y > MIN_DROP_DIST) {
                float pixel_velocity = (float)dist_y / dt;
                
                // 상대 속도 = (픽셀 속도) / (사람 키)
                // 키가 클수록(가까울수록) 픽셀 변화량이 크므로 키로 나누어 보정
                if (bh > 0) current_rel_vel = pixel_velocity / (float)bh;
            }
        }
        
        // 상태 저장
        prev_cy = cy;
        prev_time_ms = now;

        // [로직 C] 이벤트 트리거 (급격한 하강)
        // 엣지에 의해 무시되지 않았고(정상 화면) && 속도가 임계값(1.2)을 넘었을 때
        if (!ignore_by_edge && current_rel_vel > REL_VEL_TH) {
            was_rapid_drop = true;  // "충격 발생!" 기억
            rapid_drop_ts = now;    // 발생 시간 기록
        }

        // [로직 D] 이벤트 만료
        // 충격 발생 후 2.5초가 지날 때까지 눕지 않으면 이벤트 취소
        if (was_rapid_drop && (now - rapid_drop_ts > IMPACT_WINDOW_MS)) {
            was_rapid_drop = false;
        }

        float areaFrac = (float)(bw * bh) / (float)(im_w * im_h);
        float ar = (float)bh / (float)bw; 

        // [로직 E] 자세(비율) 판단
        // 박스가 충분히 크고 && 엣지에 무시되지 않았을 때
        if (areaFrac >= MIN_BOX_AREA_FRAC) {
            if (!ignore_by_edge && ar < FALL_AR_TH) {
                is_lying_now = true; // 현재 누워있음
            }
        }

        // [로직 F] 최종 낙상 확정 (Sequence Check)
        // (과거에 충격 있었음) AND (현재 누워있음)
        if (was_rapid_drop && is_lying_now) {
            // 지속 시간 측정 시작
            if (lying_start_ms == 0) lying_start_ms = now;
            
            uint32_t duration = now - lying_start_ms;
            // 1초 이상 유지되면 낙상 확정
            if (duration >= (uint32_t)(FALL_HOLD_SEC * 1000.0f)) {
                fall_confirmed = true;
            }
        } else {
            // 조건 불충족 시 타이머 리셋
            lying_start_ms = 0;
            
            // 회복 판단: 다시 서있는 비율(1.2 이상)이 되면 모든 경보 해제
            if (ar > 1.2f) { 
                fall_confirmed = false;
                fall_alert_sent = false;
                was_rapid_drop = false; 
            }
        }

    } else {
        // 사람이 없으면 모든 추적 정보 리셋
        prev_cy = -1;
        current_rel_vel = 0.0f;
        lying_start_ms = 0;
    }

    // ---------------------------------------------------------
    // STEP 3: OSD 그리기 (상태 시각화)
    // ---------------------------------------------------------
    if (ObjDet.getResultCount() > 0) {
        for (int i = 0; i < ObjDet.getResultCount(); i++) {
            int obj_type = results[i].type();
            if (!itemList[obj_type].filter) continue;
            ObjectDetectionResult item = results[i];
            
            int xmin = (int)(item.xMin() * im_w);
            int xmax = (int)(item.xMax() * im_w);
            int ymin = (int)(item.yMin() * im_h);
            int ymax = (int)(item.yMax() * im_h);

            if (strcmp(itemList[obj_type].objectName, "person") == 0) {
                // 낙상 확정 상태 (Red Box)
                if (fall_confirmed && i == bestPersonIdx) {
                    OSD.drawRect(CHANNEL_OSD, xmin, ymin, xmax, ymax, 4, OSD_COLOR_RED);
                    OSD.drawText(CHANNEL_OSD, xmin, ymin - 20, "FALL CONFIRMED", OSD_COLOR_RED);
                    
                    // 시리얼 알림 (1회)
                    if (!fall_alert_sent) {
                        Serial.println("Fall");
                        fall_alert_sent = true;
                    }
                } 
                // 감지 중인 메인 대상
                else if (i == bestPersonIdx) {
                    char text_str[64];
                    
                    // 케이스 1: 화면 꽉 참 (판단 불가 - Blue Box)
                    if (ignore_by_edge) {
                        OSD.drawRect(CHANNEL_OSD, xmin, ymin, xmax, ymax, 3, OSD_COLOR_BLUE);
                        snprintf(text_str, sizeof(text_str), "Full-Screen Ignored");
                        OSD.drawText(CHANNEL_OSD, xmin, ymin - 20, text_str, OSD_COLOR_BLUE);
                    } 
                    // 케이스 2: 충격 감지됨 (주의 - Yellow Box)
                    else if (was_rapid_drop) {
                        OSD.drawRect(CHANNEL_OSD, xmin, ymin, xmax, ymax, 3, OSD_COLOR_YELLOW);
                        snprintf(text_str, sizeof(text_str), "Check RV %.2f", current_rel_vel);
                        OSD.drawText(CHANNEL_OSD, xmin, ymin - 20, text_str, OSD_COLOR_YELLOW);
                    } 
                    // 케이스 3: 정상 상태 (Green Box)
                    else {
                        OSD.drawRect(CHANNEL_OSD, xmin, ymin, xmax, ymax, 3, OSD_COLOR_GREEN);
                        snprintf(text_str, sizeof(text_str), "OK RV %.2f", current_rel_vel);
                        OSD.drawText(CHANNEL_OSD, xmin, ymin - 20, text_str, OSD_COLOR_GREEN);
                    }
                } else {
                    // 메인 대상이 아닌 주변인 (White Box)
                    OSD.drawRect(CHANNEL_OSD, xmin, ymin, xmax, ymax, 3, OSD_COLOR_WHITE);
                }
            } else {
                // 사람이 아닌 물체 (White Box)
               // OSD.drawRect(CHANNEL_OSD, xmin, ymin, xmax, ymax, 3, OSD_COLOR_WHITE);
            }
        }
    }

    OSD.update(CHANNEL_OSD);
    delay(100); 
}