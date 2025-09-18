# -*- coding: utf-8 -*-
# 실행 전: pip install google-api-python-client
# 실행: python collect_playlist_titles.py

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import datetime, re, os, sys, time, traceback, json

# ================== 필수 설정 ==================
API_KEY = "AIzaSyBYUKxiQxR0zZXJXBiBZTLssmdeR_uYjU8"  # ← 사용자 API 키
# ==============================================

# -------- 수집 조건 --------
MONTHS = 1                     # 최근 n개월
MIN_VIEWS = 10_000             # 조회수 하한
MIN_DURATION = 60 * 60         # 1시간
MAX_DURATION = 3 * 60 * 60     # 3시간
MAX_CHANNEL_SUBS = 50_000      # 구독자 수 상한(대형 채널 배제 보조)
MIN_VIEWS_PER_SUB = 1.5        # 조회수/구독자 효율 하한
MIN_VPD = 3_000                # 일평균 조회수 하한
MIN_LIKE_RATE = 0.008          # 좋아요/조회수 0.8%+
MIN_COMMENT_RATE = 0.0005      # 댓글/조회수 0.05%+
POOL_LIMIT = 3000              # 내부 풀 상한(성능/쿼터 보호)
OUTPUT_TOP_N = 1000            # 저장 상위 N개
OUTPUT_PATH = "smart_titles_top.txt"
OUTPUT_DATE_FORMAT = "%Y%m%d"   # 신규 결과 파일 날짜 포맷
STATE_PATH = "search_state.json"  # 키워드별 페이지 진행 상태 저장
ORDER = "viewCount"            # "date" 가능
RETRY_MAX = 5                   # 재시도 횟수
RETRY_BACKOFF = 1.6             # 지수 백오프 계수
PAGE_LIMIT = 10                  # 키워드별 검색 페이지 제한(호출량 절감)
PAGE_EXPAND_STEP = 5            # 신규 항목 없을 때 추가 탐색 페이지 수
PAGE_EXPAND_MAX = 10            # 최대 확장 횟수(무한 루프 방지)
CALL_DELAY = 0.2                # 연속 호출 간 지연(초)

# YouTube API 쿼터 사용 계획(대략치). 환경변수로 조정 가능
# 참고: search.list ≈ 100, videos.list ≈ 1, channels.list ≈ 1
QUOTA_BUDGET = int(os.getenv("YT_QUOTA_BUDGET", "9000"))   # 오늘 사용할 목표 유닛
QUOTA_SAFETY = int(os.getenv("YT_QUOTA_SAFETY", "50"))     # 남겨둘 안전 마진
COST_SEARCH = 100
COST_VIDEOS = 1
COST_CHANNELS = 1
PER_PAGE_COST = COST_SEARCH + COST_VIDEOS + COST_CHANNELS  # 키워드당 한 페이지 처리 예상 비용

# 검색 키워드: 요청에 따라 축소
TARGET_KEYWORDS = ["플레이리스트", "playlist"]
# 제목 필수 포함 키워드: 제목에 아래 중 하나가 반드시 포함되어야 함
REQUIRED_TITLE_KEYWORDS = ["플레이리스트", "playlist"]

# 20·30대 연령층 가중을 위한 제목 힌트(검색엔 사용하지 않음, 스코어 가중에만 반영)
AGE_HINTS = [
    "퇴근", "출근", "드라이브", "공부", "집중", "운동", "카페", "브금",
    "감성", "여유", "기분전환", "힙", "rnb", "알앤비", "lofi", "groove", "urban", "chill", "focus", "study", "work"
]

# 공식/대형 음악 유통 레이블/채널 제외(“구독자빨” 완화)
EXCLUDE_CHANNEL_PATTERNS = re.compile(r"(official|topic|vevo|1thek|warner|universal|sony\s*music)", re.I)

def log(msg): print(msg, flush=True)

def yt_call(fn, **kwargs):
    delay = 1.0
    for attempt in range(1, RETRY_MAX + 1):
        try:
            return fn(**kwargs).execute()
        except HttpError as e:
            code = getattr(e.resp, "status", None)
            if code == 403:
                log(f"[ERROR] API 403 응답 감지 → 즉시 종료")
                sys.exit(1)
            if code in (429, 500, 503):
                log(f"[WARN] API 오류(code={code}) 재시도 {attempt}/{RETRY_MAX} 후 {delay:.1f}s 대기")
                time.sleep(delay); delay *= RETRY_BACKOFF
                continue
            else:
                log(f"[ERROR] HttpError(code={code}): {e}")
                raise
        except Exception as e:
            log(f"[ERROR] 예외 발생: {e}")
            if attempt == RETRY_MAX: raise
            time.sleep(delay); delay *= RETRY_BACKOFF
    raise RuntimeError("API 호출 재시도 초과")

def build_client():
    if not API_KEY or API_KEY == "YOUR_API_KEY":
        print("[ERROR] API_KEY 미설정", file=sys.stderr); sys.exit(1)
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
    # 20·30대 가중 휴리스틱: 길이/구분자/숫자/AGE_HINTS
    score = 0.0
    t = (title or "").strip()
    L = len(t)
    if 18 <= L <= 42: score += 1.2
    if re.search(r'[·|\|\•]', t): score += 0.5
    if re.search(r'\d', t): score += 0.3
    if any(k in t.lower() for k in [h.lower() for h in AGE_HINTS]): score += 0.9
    return score

def title_has_required_keywords(title: str) -> bool:
    t = (title or "").lower()
    return ("플레이리스트" in t) or ("playlist" in t)

def search(youtube, query, published_after, page_token=None, max_results=50):
    return yt_call(
        youtube.search().list,
        q=query, part="snippet", type="video", order=ORDER,
        publishedAfter=published_after, maxResults=max_results, pageToken=page_token
    )

def fetch_video_details(youtube, ids):
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
    ts = (snippet or {}).get("publishedAt")
    if not ts:
        return datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=1)
    try:
        # RFC3339 'Z'를 timezone-aware 형태로 변환
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

def build_dated_output_path(base_path, dt=None):
    """기본 경로에 날짜를 붙인 새 파일 경로를 만든다."""
    dt = dt or datetime.datetime.now()
    root, ext = os.path.splitext(base_path)
    suffix = dt.strftime(OUTPUT_DATE_FORMAT)
    return f"{root}_{suffix}{ext}"

def append_unique_lines(path, lines, encoding="utf-8"):
    """중복을 제거한 뒤 신규 라인을 마스터 파일에 추가하고, 날짜가 붙은 파일을 생성한다."""
    existing = set()
    if os.path.exists(path):
        with open(path, "r", encoding=encoding) as f:
            for line in f:
                existing.add(line.rstrip("\n"))
    candidates = [ln.strip() for ln in lines if (ln or "").strip()]
    new_items = [ln for ln in candidates if ln not in existing]
    dated_path = None
    if new_items:
        with open(path, "a", encoding=encoding) as f:
            for ln in new_items:
                f.write(ln + "\n")
        dated_path = build_dated_output_path(path)
        with open(dated_path, "w", encoding=encoding) as f:
            for ln in new_items:
                f.write(ln + "\n")
    return len(new_items), len(candidates) - len(new_items), len(existing), dated_path

def load_state(path=STATE_PATH):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_state(state, path=STATE_PATH):
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)

def collect_titles(page_limit=None, allow_expand=True, expand_attempts=0):
    youtube = build_client()
    now = datetime.datetime.now(datetime.UTC)
    published_after = (now - datetime.timedelta(days=30*MONTHS)).isoformat().replace("+00:00", "Z")
    pool = []
    budget = QUOTA_BUDGET
    state = load_state()
    page_limit_value = page_limit or PAGE_LIMIT
    log(f"[INFO] 시작 budget={budget}, safety={QUOTA_SAFETY}, page_limit={page_limit_value}")

    quota_hit_global = False
    quota_error_detected = False

    for idx, kw in enumerate(TARGET_KEYWORDS, 1):
        log(f"[{idx}/{len(TARGET_KEYWORDS)}] 검색 키워드: {kw}")
        kw_state = state.get(kw, {}) or {}
        skip_pages = int(kw_state.get("page_cursor", 0) or 0)
        if skip_pages:
            log(f"[INFO] {kw} 검색 시작 전에 {skip_pages}페이지 건너뜀")

        remaining_keywords = len(TARGET_KEYWORDS) - idx + 1
        allocatable = max(0, budget - QUOTA_SAFETY)
        keyword_budget_total = allocatable // remaining_keywords if remaining_keywords else allocatable
        skip_cost = skip_pages * COST_SEARCH
        if keyword_budget_total <= skip_cost:
            log(f"[INFO] {kw}는 예산 부족으로 대기 (현재 예산 {keyword_budget_total}, skip 비용 {skip_cost})")
            state[kw] = {"page_cursor": skip_pages}
            quota_hit_global = True
            continue
        if PER_PAGE_COST > 0:
            page_budget_pages = (keyword_budget_total - skip_cost) // PER_PAGE_COST
        else:
            page_budget_pages = page_limit_value
        page_limit_for_run = min(page_limit_value, page_budget_pages)
        if page_limit_for_run <= 0:
            log(f"[INFO] {kw} 새 페이지 처리 예산 부족 -> 다음 키워드로 이동")
            state[kw] = {"page_cursor": skip_pages}
            quota_hit_global = True
            continue
        log(f"[INFO] {kw} 키워드 예산 {keyword_budget_total}유닛, 이번 실행 최대 {page_limit_for_run}페이지 처리")

        skipped = 0
        processed_pages = 0
        kw_cursor = skip_pages
        next_token = None
        quota_hit = False
        state_updated = False
        kw_budget_left = keyword_budget_total
        while True:
            if kw_budget_left < COST_SEARCH:
                log(f"[INFO] {kw} 잔여 예산 부족으로 중단 (남은 예산 {kw_budget_left})")
                state[kw] = {"page_cursor": kw_cursor}
                state_updated = True
                quota_hit_global = True
                break
            # 예산 점검: search 호출 전 여유 확인
            if budget < (COST_SEARCH + QUOTA_SAFETY):
                log(f"[INFO] budget 부족으로 search 중단 (remain={budget})")
                quota_hit = True
                quota_hit_global = True
                break
            # search.list 호출: 쿼터/레이트 제한 시 부분 결과 저장 후 중단
            try:
                s = search(youtube, kw, published_after, page_token=next_token)
            except HttpError as e:
                code = getattr(e.resp, "status", None)
                if code in (403, 429):
                    log(f"[WARN] search 제한(code={code}). 현재까지 수집 {len(pool)}개, 키워드 루프 중단")
                    quota_hit = True
                    quota_hit_global = True
                    quota_error_detected = True
                    break
                raise
            budget -= COST_SEARCH
            kw_budget_left -= COST_SEARCH
            if skip_pages and skipped < skip_pages:
                skipped += 1
                next_token = s.get("nextPageToken")
                if not next_token:
                    state[kw] = {"page_cursor": 0}
                    state_updated = True
                    break
                continue

            items = s.get("items", [])
            if not items:
                state[kw] = {"page_cursor": 0}
                state_updated = True
                break

            processed_pages += 1
            kw_cursor += 1

            video_ids = [i.get("id", {}).get("videoId") for i in items if i.get("id", {}).get("videoId")]
            if kw_budget_left < (COST_VIDEOS + COST_CHANNELS):
                log(f"[INFO] {kw} 잔여 예산으로 상세 정보 조회 불가 (남은 예산 {kw_budget_left})")
                state[kw] = {"page_cursor": kw_cursor}
                state_updated = True
                quota_hit_global = True
                break

            time.sleep(CALL_DELAY)
            try:
                # 예산 점검: videos 호출 비용
                if budget < (COST_VIDEOS + QUOTA_SAFETY):
                    log(f"[INFO] budget 부족으로 videos 중단 (remain={budget})")
                    quota_hit = True
                    quota_hit_global = True
                    break
                v = fetch_video_details(youtube, video_ids)
            except HttpError as e:
                code = getattr(e.resp, "status", None)
                if code in (403, 429):
                    log(f"[WARN] videos 제한(code={code}). 현재까지 수집 {len(pool)}개, 키워드 루프 중단")
                    quota_hit = True
                    quota_hit_global = True
                    quota_error_detected = True
                    break
                raise
            budget -= COST_VIDEOS
            kw_budget_left -= COST_VIDEOS
            videos = v.get("items", [])
            channel_ids = list({(it.get("snippet") or {}).get("channelId") for it in videos if (it.get("snippet") or {}).get("channelId")})
            if kw_budget_left < COST_CHANNELS:
                log(f"[INFO] {kw} 잔여 예산으로 channel 조회 불가 (남은 예산 {kw_budget_left})")
                state[kw] = {"page_cursor": kw_cursor}
                state_updated = True
                quota_hit_global = True
                break
            time.sleep(CALL_DELAY)
            try:
                # 예산 점검: channels 호출 비용
                if budget < (COST_CHANNELS + QUOTA_SAFETY):
                    log(f"[INFO] budget 부족으로 channels 중단 (remain={budget})")
                    quota_hit = True
                    quota_hit_global = True
                    break
                ch_map = fetch_channel_subs(youtube, channel_ids)
            except HttpError as e:
                code = getattr(e.resp, "status", None)
                if code in (403, 429):
                    log(f"[WARN] channels 제한(code={code}). 현재까지 수집 {len(pool)}개, 키워드 루프 중단")
                    quota_hit = True
                    quota_hit_global = True
                    quota_error_detected = True
                    break
                raise
            budget -= COST_CHANNELS
            kw_budget_left -= COST_CHANNELS

            for it in videos:
                sn = it.get("snippet", {}) or {}
                st = it.get("statistics", {}) or {}
                cd = it.get("contentDetails", {}) or {}

                title = (sn.get("title") or "").strip()
                ch_id = sn.get("channelId", "") or ""
                ch_info = ch_map.get(ch_id, {}) or {}
                ch_title = ch_info.get("channel_title", "") or ""
                subs = ch_info.get("subs", 0) or 0

                # 공식/대형 레이블 제외
                if EXCLUDE_CHANNEL_PATTERNS.search(ch_title):
                    continue

                # 제목 필수 키워드 포함
                if not title_has_required_keywords(title):
                    continue
                # 한글 제목 + 조회수
                if not is_korean_title(title): 
                    continue
                views = int(st.get("viewCount", 0) or 0)
                if views < MIN_VIEWS: 
                    continue

                # 길이: 1~3시간
                dur = parse_duration_iso8601(cd.get("duration"))
                if not (MIN_DURATION <= dur <= MAX_DURATION):
                    continue

                # 대형채널 방어
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
                score = ol * (1 + tqs * 0.25)  # 20·30대 힌트 가중 약간 상향

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
            if not next_token:
                state[kw] = {"page_cursor": 0}
                state_updated = True
                break
            if len(pool) >= POOL_LIMIT:
                state[kw] = {"page_cursor": kw_cursor}
                state_updated = True
                break
            if processed_pages >= page_limit_for_run:
                state[kw] = {"page_cursor": kw_cursor}
                state_updated = True
                break

        if not state_updated:
            state[kw] = {"page_cursor": kw_cursor}

        if quota_hit:
            break

        if len(pool) >= POOL_LIMIT:
            log(f"[INFO] POOL_LIMIT({POOL_LIMIT}) 도달, 다음 키워드 중단")
            break

    save_state(state)

    # 중복 제거 + 점수순 정렬
    pool.sort(key=lambda x: x["score"], reverse=True)
    titles, seen = [], set()
    for row in pool:
        t = row["title"]
        if t not in seen:
            seen.add(t)
            titles.append(t)

    # 기존 파일 보존 + 신규만 추가
    to_write = titles[:OUTPUT_TOP_N]
    added, dup, prev, dated_path = append_unique_lines(OUTPUT_PATH, to_write)
    if dated_path:
        log(f"[DONE] Appended {added} (dup={dup}) to {OUTPUT_PATH}. prev={prev}, pool={len(pool)}, new_file={dated_path}")
    else:
        log(f"[DONE] 신규 항목 없음 (dup={dup}). prev={prev}, pool={len(pool)}")

    if allow_expand and added == 0 and not quota_error_detected and not quota_hit_global:
        if expand_attempts >= PAGE_EXPAND_MAX:
            log(f"[INFO] 신규 항목 없고 확장 한도({PAGE_EXPAND_MAX}) 도달 → 추가 확장 중단")
        else:
            next_limit = page_limit_value + PAGE_EXPAND_STEP
            log(f"[INFO] 신규 항목 없음 → page_limit {page_limit_value} → {next_limit} 재시도 (attempt {expand_attempts + 1}/{PAGE_EXPAND_MAX})")
            collect_titles(page_limit=next_limit, allow_expand=True, expand_attempts=expand_attempts + 1)

if __name__ == "__main__":
    try:
        collect_titles()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
