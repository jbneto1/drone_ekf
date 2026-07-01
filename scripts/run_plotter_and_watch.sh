#!/usr/bin/env bash

status_file="$(mktemp)"

cleanup() {
    rm -f "${status_file}"
}

trap cleanup EXIT

export DRONE_EKF_PLOTTER_STATUS_FILE="${status_file}"
roslaunch drone_ekf plotter.launch
launch_status=$?
unset DRONE_EKF_PLOTTER_STATUS_FILE

echo ""
echo "Plotter launch exited."

if [[ ! -s "${status_file}" ]]; then
    echo "The plotter node did not initialize its saver status."
    echo "Check the roslaunch output above for a startup error."
    exit "${launch_status}"
fi

mapfile -t saver_status < "${status_file}"
state="${saver_status[0]:-unknown}"
saver_log="${saver_status[1]:-}"
saver_pid="${saver_status[2]:-}"
saver_error="${saver_status[3]:-}"

case "${state}" in
    no_data)
        echo "The plotter received no data, so no plots were generated."
        exit 0
        ;;
    failed)
        echo "The plot saver failed: ${saver_error:-unknown error}"
        if [[ -n "${saver_log}" && -f "${saver_log}" ]]; then
            echo ""
            cat -- "${saver_log}"
        fi
        exit 1
        ;;
    complete)
        echo "Plot saving is complete."
        if [[ -n "${saver_log}" && -f "${saver_log}" ]]; then
            echo ""
            cat -- "${saver_log}"
        fi
        exit 0
        ;;
    detached_started)
        ;;
    *)
        echo "Unexpected plotter saver state: ${state}"
        exit "${launch_status}"
        ;;
esac

echo "Watching detached saver log:"
echo "  ${saver_log}"
if [[ -n "${saver_pid}" ]]; then
    echo "Detached saver PID: ${saver_pid}"
fi
echo "Press Ctrl+C to stop watching."
echo ""

# -F retries until the detached child creates the file and follows it by name.
tail -n +1 -F -- "${saver_log}"
