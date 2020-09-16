# 신호등 신호 변경, 남은 시간 출력, 시간 연장(10초)
# detect 안에서 돌아가야함
# Signal 클래스는 1초에 한번 돌아감
# 중간에 사람이 나가면 대기시간 0초로 변경

def Search(x, arr):
    for i in range(1, len(arr)):
        if arr[i - 1][0] <= x <= arr[i][0]:
            return i - 1, i
    return -1, -1


def Check(x, y, point1, point2, isDown):
    x1 = point1[0]
    y1 = point1[1]
    x2 = point2[0]
    y2 = point2[1]

    slope = (y2 - y1) / (x2 - x1)
    bias = -(slope * x1) + y1
    Y = x * slope + bias

    if isDown is True and Y <= y:
        return True
    elif isDown is False and Y >= y:
        return True
    else:
        return False


def areaDetection(now, up_arr, dn_arr):  # 영역안에 있다면 true, else false

    cnt = 0
    for i in range(len(now)):
        x = now[i][0]
        y = now[i][1]
        if up_arr[0][0] > x > up_arr[-1][0]:  # x가 영역을 벗어났다면
            continue
        upArrLowerIdx, upArrUpperIdx = Search(x, up_arr)
        dnArrLowerIdx, dnArrUpperIdx = Search(x, dn_arr)
        if upArrLowerIdx == -1 or dnArrLowerIdx == -1:
            continue

        isdown = Check(x, y, up_arr[upArrLowerIdx], up_arr[upArrUpperIdx], True)
        isup = Check(x, y, dn_arr[dnArrLowerIdx], dn_arr[dnArrUpperIdx], False)
        if isdown is True and isup is True:
            cnt += 1
    return cnt


def is_people_in(current_signal, xy, up_arr_0,dn_arr_0, up_arr_1, dn_arr_1):
    if current_signal == 0:
        up_arr, dn_arr = up_arr_0, dn_arr_0
    else:
        up_arr, dn_arr = up_arr_1, dn_arr_1

    output = areaDetection(xy, up_arr, dn_arr)
    signal_string = 'RED' if current_signal == 0 else 'GREEN'
    place_string = '신호대기 영역' if current_signal == 0 else '횡단보도 영역'
    print("[{}]{}: {}명 탐지".format(signal_string, place_string, output))

    return output


class Signal:
    def __init__(self, people_exist=10, walk_time=20, wait=5, extension=10, max_extension=2):
        '''

        :param people_exist: 최소 대기 시간
        :param walk_time:    최대 보행 시간
        :param wait:         연속 신호 켜짐 방지
        :param extension:    신호 연장 시간
        '''

        self.FIRST = 0
        self.people_exist = people_exist
        self.walk_time = walk_time
        self.wait = wait
        self.extension = extension
        self.max_extension = max_extension

        self.count = 0  # 대기 시간
        self.remain_time = 0  # 남은 보행 시간
        self.wait_count = 0  # 연속 신호 켜짐 방지 시간 카운트
        self.signal = 0  # 신호
        self.extension_count = 0 # 신호 연장 횟수

    def get_signal(self):
        return self.signal

    def get_remain_time(self):
        return self.remain_time

    def change_to_green(self):
        # 빨간불일때 최소 차량 통행 시간을 넘겼는지, 사림이 15초 이상 대기중인지 확인하고 신호 변경
        if self.count >= self.people_exist:
            self.signal = 1
            self.remain_time = self.walk_time - 1

    def change_to_red(self, output):
        # 파란불일때 남은 시간이 10초 이하 2초 이상이고 사람이 있으면 신호 연장. 신호 끝나면 다시 빨간불로 변경
        if 2 <= self.remain_time < 7:
            if output and self.extension_count < self.max_extension:
                self.remain_time += self.extension
                self.extension_count += 1

        elif self.remain_time <= 0:
            self.signal = 0
            self.wait_count = 0
            self.count = 0
            self.extension_count = 0

    def __call__(self, xy, up_arr_0,dn_arr_0, up_arr_1, dn_arr_1):
        output = is_people_in(self.signal, xy, up_arr_0,dn_arr_0, up_arr_1, dn_arr_1)

        if self.signal:  # 초록불
            self.remain_time -= 1

            self.change_to_red(output)

        elif not self.signal:  # 빨간불
            # 신호 변경
            if self.FIRST == 0:
                self.wait_count += 1000
                self.FIRST = 1
            if output and self.wait_count >= self.wait:
                self.count += 1

                self.change_to_green()

            # 중간에 사람이 나가면 초기화
            elif not output and self.count != 0:
                self.count = 0

            else:
                self.wait_count += 1
