# -*- coding: utf-8 -*-
# 실행: pip install google-api-python-client && python collect_playlist_titles.py

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import datetime, re, os, sys, time, traceback

# ================== 설정 ==================
API_KEY = os.getenv("YT_API_KEY", "AIzaSyBYUKxiQxR0zZXJXBiBZTLssmdeR_uYjU8")  # 환경변수 우선
MONTHS = 1                     # 최근 n개월
MIN_VIEWS = 10_000             # 조회수 하한
MIN_DURATION = 20 * 60         # 20분(쇼츠/짧은 영상 제외)
MAX_DURATION = 2 * 60 * 60     # 2시간(초장문 스트리밍 제외)
MAX_CHANNEL_SUBS = 50_000      # 이 값 초과 채널은 효율 검증 필요
MIN_VIEWS_PER_SUB = 1.5        # 조회수/구독자 효율 하한(대형 채널 방어)
MIN_VPD = 3_000                # 일평균 조회수 하한 (성장속도)
MIN_LIKE_RATE = 0.008          # 좋아요/조회수 0.8%+
MIN_COMMENT_RATE = 0.0005      # 댓글/조회수 0.05%+
POOL_LIMIT = 3000              # 내부 수집 풀 상한(쿼터/성능 보호)
OUTPUT_TOP_N = 1000            # 저장할 상위 N개 제목
OUTPUT_PATH = "smart_titles_top.txt"
ORDER = "viewCount"            # "date"로 바꿔도 됨
RETRY_MAX = 5                  # API 오류 재시도 횟수
RETRY_BACKOFF = 1.5            # 지수 백오프 배수
PAGE_LIMIT = 3                 # 키워드별 검색 페이지 제한(호출량 절감)
CALL_DELAY = 0.2               # 연속 호출 간 지연(초)

# 밤/밤무드 제거 + 범용 확장 키워드 (상황/감정/장르/글로벌)
TARGET_KEYWORDS = [
    # 상황/활동
    "플리", "퇴근길", "출근길", "드라이브", "운동", "공부", "여행", "카페", "브금", "집중",
    # 감정/효용
    "감성", "우울할때", "행복", "사랑", "연애", "설렘", "힐링", "편안한", "여유", "기분전환", "스트레스",
    # 장르/무드
    "힙합", "알앤비", "rnb", "trap", "afrobeat", "lofi", "indie", "soul", "groove", "pop", "mellow", "chill", "chillout", "urban",
    # 글로벌 일반 검색어
    "playlist", "vibe", "mood", "mix", "focus", "study", "work"
]

# 공식/대형 뮤직 채널 제외 패턴
EXCLUDE_CHANNEL_PATTERNS = re.compile(r"(official|topic|vevo|1thek|warner|universal|sony\s*music)", re.I)
# =========================================

def log(msg): print(msg, flush=True)

def yt_call(fn, **kwargs):
    """YouTube API 호출 with 재시도/백오프"""
    delay = 1.0
    for attempt in range(1, RETRY_MAX + 1):
        try:
            return fn(**kwargs).execute()
        except HttpError as e:
            code = getattr(e.resp, "status", None)
            if code in (403, 429, 500, 503):
                log(f"[WARN] API 오류(code={code}) 재시도 {attempt}/{RETRY_MAX} 후 {delay:.1f}s 대기")
                time.sleep(delay)
                delay *= RETRY_BACKOFF
                continue
            else:
                log(f"[ERROR] HttpError(code={code}): {e}")
                raise
        except Exception as e:
            log(f"[ERROR] 예외 발생: {e}")
            if attempt == RETRY_MAX:
                raise
            time.sleep(delay)
            delay *= RETRY_BACKOFF
    raise RuntimeError("API 호출 재시도 초과")

def build_client():
    if not API_KEY or API_KEY == "YOUR_API_KEY":
        print("[ERROR] API_KEY 미설정", file=sys.stderr)
        sys.exit(1)
    return build('youtube', 'v3', developerKey=API_KEY)

def parse_duration_iso8601(iso):
    if not iso: return 0
    h = m = s = 0
    H = re.search(r'(\d+)H', iso); M = re.search(r'(\d+)M', iso); S = re.search(r'(\d+)S', iso)
    if H: h = int(H.group(1))
    if M: m = int(M.group(1))
    if S: s = int(S.group(1))
    return h*3600 + m*60 + s

def is_korean_title(title):
    return bool(re.search('[가-힣]', title or ""))

def title_quality_score(title):
    score = 0.0
    t = (title or "").strip()
    L = len(t)
    if 18 <= L <= 42: score += 1.2
    if re.search(r'[·|\|\•]', t): score += 0.5
    if re.search(r'\d', t): score += 0.3
    keywords = ["퇴근","출근","집중","드라이브","비","새벽","감성","힙","chill","플리","카페","운동","공부","여행","브금","기분전환"]
    if any(k in t for k in keywords): score += 0.7
    return score

def search(youtube, query, published_after, page_token=None, max_results=50):
    return yt_call(
        youtube.search().list,
        q=query, part="snippet", type="video", order=ORDER,
        publishedAfter=published_after, maxResults=max_results, pageToken=page_token
    )

def fetch_video_details(youtube, ids):
    # ids가 비면 API 400 에러 → 사전 차단
    if not ids: return {"items": []}
    return yt_call(
        youtube.videos().list,
        part="snippet,contentDetails,statistics", id=",".join(ids)
    )

def fetch_channel_subs(youtube, channel_ids):
    if not channel_ids: return {}
    resp = yt_call(
        youtube.channels().list,
        part="statistics,snippet", id=",".join(channel_ids)
    )
    info = {}
    for c in resp.get("items", []):
        stats = c.get("statistics", {}) or {}
        subs = 0
        if not stats.get("hiddenSubscriberCount"):
            subs = int(stats.get("subscriberCount", 0) or 0)
        title = c.get("snippet", {}).get("title", "") or ""
        info[c.get("id")] = {"subs": subs, "channel_title": title}
    return info

def safe_published_at(snippet):
    # RFC3339 → timezone-aware(UTC) datetime로 변환
    ts = (snippet or {}).get("publishedAt")
    if not ts:
        return datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=1)
    try:
        # 'Z' → '+00:00' 치환 후 fromisoformat
        return datetime.datetime.fromisoformat(ts.replace("Z","+00:00"))
    except Exception:
        return datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=1)

def organic_lift(views, published_at_dt, likes, comments, subs):
    days = max((datetime.datetime.now(datetime.UTC) - published_at_dt).days, 1)
    vpd = views / days
    like_rate = (likes / views) if views else 0.0
    comment_rate = (comments / views) if views else 0.0
    engagement = 0.6*like_rate + 0.4*comment_rate
    denom = (subs ** 0.3) if subs > 0 else 1.0
    return vpd * engagement / denom, vpd, like_rate, comment_rate

def append_unique_lines(path, lines, encoding="utf-8"):
    """파일에 중복 없이 신규 라인만 추가. 반환값: (추가개수, 중복개수, 기존총개수)"""
    existing = set()
    if os.path.exists(path):
        with open(path, "r", encoding=encoding) as f:
            for line in f:
                existing.add(line.rstrip("\n"))
    candidates = [ln.strip() for ln in lines if (ln or "").strip()]
    new_items = [ln for ln in candidates if ln not in existing]
    if new_items:
        with open(path, "a", encoding=encoding) as f:
            for ln in new_items:
                f.write(ln + "\n")
    return len(new_items), len(candidates) - len(new_items), len(existing)

def collect_titles():
    youtube = build_client()
    now = datetime.datetime.now(datetime.UTC)
    published_after = (now - datetime.timedelta(days=30*MONTHS)).isoformat().replace("+00:00", "Z")
    pool = []

    for idx, kw in enumerate(TARGET_KEYWORDS, 1):
        log(f"[{idx}/{len(TARGET_KEYWORDS)}] 검색 키워드: {kw}")
        next_token = None
        pages = 0
        quota_hit = False
        while True:
            try:
                s = search(youtube, kw, published_after, page_token=next_token)
            except HttpError as e:
                code = getattr(e.resp, "status", None)
                if code in (403, 429):
                    log(f"[WARN] 쿼터/레이트 제한(code={code})로 현재까지 수집분만 반환")
                    quota_hit = True
                    break
                raise
            items = s.get("items", [])
            if not items:
                break

            video_ids = [i.get("id", {}).get("videoId") for i in items if i.get("id", {}).get("videoId")]
            time.sleep(CALL_DELAY)
            try:
                v = fetch_video_details(youtube, video_ids)
            except HttpError as e:
                code = getattr(e.resp, "status", None)
                if code in (403, 429):
                    log(f"[WARN] videos 제한(code={code})로 중단, 부분 결과 저장")
                    quota_hit = True
                    break
                raise
            videos = v.get("items", [])
            channel_ids = list({(it.get("snippet") or {}).get("channelId") for it in videos if (it.get("snippet") or {}).get("channelId")})
            time.sleep(CALL_DELAY)
            try:
                ch_map = fetch_channel_subs(youtube, channel_ids)
            except HttpError as e:
                code = getattr(e.resp, "status", None)
                if code in (403, 429):
                    log(f"[WARN] channels 제한(code={code})로 중단, 부분 결과 저장")
                    quota_hit = True
                    break
                raise

            for it in videos:
                sn = it.get("snippet", {}) or {}
                st = it.get("statistics", {}) or {}
                cd = it.get("contentDetails", {}) or {}
                title = (sn.get("title") or "").strip()
                ch_id = sn.get("channelId", "") or ""
                ch_info = ch_map.get(ch_id, {}) or {}
                ch_title = ch_info.get("channel_title", "") or ""
                subs = ch_info.get("subs", 0) or 0

                # 채널 제외(공식/대형 레이블)
                if EXCLUDE_CHANNEL_PATTERNS.search(ch_title):
                    continue

                # 필수 조건
                if not is_korean_title(title):
                    continue
                views = int(st.get("viewCount", 0) or 0)
                if views < MIN_VIEWS:
                    continue

                dur = parse_duration_iso8601(cd.get("duration"))
                if not (MIN_DURATION <= dur <= MAX_DURATION):
                    continue

                # 대형채널 방어: 구독자수 기준 + 효율 기준
                if subs > MAX_CHANNEL_SUBS and (views / max(subs, 1)) < MIN_VIEWS_PER_SUB:
                    continue

                likes = int(st.get("likeCount", 0) or 0)
                comments = int(st.get("commentCount", 0) or 0)

                published_dt = safe_published_at(sn)
                ol, vpd, lr, cr = organic_lift(views, published_dt, likes, comments, subs)

                # 반응률/성장속도 하한
                if lr < MIN_LIKE_RATE or cr < MIN_COMMENT_RATE:
                    continue
                if vpd < MIN_VPD:
                    continue

                tqs = title_quality_score(title)
                score = ol * (1 + tqs * 0.2)  # 제목 휴리스틱 가중

                pool.append({
                    "title": title,
                    "score": score,
                    "views": views,
                    "subs": subs,
                    "vpd": int(vpd),
                    "lr": round(lr, 4),
                    "cr": round(cr, 4),
                    "channel": ch_title
                })

            next_token = s.get("nextPageToken")
            pages += 1
            if not next_token or len(pool) >= POOL_LIMIT or pages >= PAGE_LIMIT:
                break
        if quota_hit:
            break
        if len(pool) >= POOL_LIMIT:
            log(f"[INFO] POOL_LIMIT({POOL_LIMIT}) 도달, 다음 키워드 중단")
            break

    # 중복 제거 + 점수순 정렬
    pool.sort(key=lambda x: x["score"], reverse=True)
    titles, seen = [], set()
    for row in pool:
        t = row["title"]
        if t not in seen:
            seen.add(t)
            titles.append(t)

    # 저장: 기존 파일 유지 + 신규만 추가(중복 방지)
    to_write = titles[:OUTPUT_TOP_N]
    added, dup, prev_total = append_unique_lines(OUTPUT_PATH, to_write)
    log(f"[DONE] Appended {added} (skipped dup={dup}) to {OUTPUT_PATH}. prev={prev_total}, pool={len(pool)}")

if __name__ == "__main__":
    try:
        collect_titles()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
