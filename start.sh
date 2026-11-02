#!/bin/bash
set -e

# 공통 함수 정의
do_down() {
    echo "docker down"
    docker compose --profile single down
    echo "cache clear"
    sync
    echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
}

do_up() {
    echo "docker up"
    docker compose --profile single up -d
}

do_log() {
    docker compose --profile single logs -f
}

# 소문자로 변환 (대소문자 무시)
ACTION="${1,,}"

case "$ACTION" in
    down)
        do_down
        ;;
    up)
        do_up
        ;;
    restart)
        do_down
        do_up
        ;;
    log|logs)
        do_log
        ;;
    *)
        echo "Usage: $0 {down|up|restart|log}"
        exit 1
        ;;
esac
