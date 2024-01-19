# 定义需求和供给
import numpy as np


class Demand(object):
    def __int__(
            self,
            id,
            req_t,
            depart_t,
            travel_t,
            arr_t,
            parking_t,
            lea_t,
            charge_label,
            char_t,
            req_interval,
            depart_interval,
            lea_interval,
            arr_interval,
            dest,
            time_label
    ):
        self.id = id
        self.req_t = req_t
        self.depart_t = depart_t
        self.travel_t = travel_t
        self.arr_t = arr_t
        self.parking_t = parking_t
        self.lea_t = lea_t
        self.charge_label = charge_label
        self.char_t = char_t
        self.req_interval = req_interval
        self.depart_interval = depart_interval
        self.lea_interval = lea_interval
        self.arr_interval = arr_interval
        self.dest = dest
        self.time_label = time_label
        self.wait_time = None
        self.queue_time = None


class ParkingLot(object):
    def __int__(
            self,
            id,
            os_num,
            os_id,
            fcs_num,
            fcs_id,
            scs_num,
            scs_id,
            dist_to_dest,
            res_fee,
            park_fee,
            fast_char_fee,
            slow_char_fee
    ):
        self.id = id
        self.os_num = os_num
        self.os_id = os_id
        self.fcs_num = fcs_id
        self.fcs_id = fcs_id
        self.scs_num = scs_num
        self.scs_id = scs_id
        self.total_slot = scs_num + os_num + fcs_num
        self.dist_to_dest = dist_to_dest
        self.res_fee = res_fee
        self.park_fee = park_fee
        self.fast_char_fee = fast_char_fee
        self.slow_char_fee = slow_char_fee

        # 需要更新的属性
        self.available_os = os_num
        self.available_fcs = fcs_num
        self.available_scs = scs_num
        self.occupancy = 0
        self.total_accept_num = 0
        self.total_accept_park_num = 0
        self.total_accept_fast_charge_num = 0
        self.total_accept_slow_charge_num = 0
        self.this_accept_num = 0
        self.this_accept_park_num = 0
        self.this_accept_fast_charge_num = 0
        self.this_accept_slow_charge_num = 0
        self.removed_park_num = 0
        self.removed_fast_charge_num = 0
        self.removed_slow_charge_num = 0

    # 每接受一个需求 调用一次该方法
    def update_accept_park_num(self):
        self.total_accept_park_num += 1
        self.this_accept_park_num += 1
        self.total_accept_num += 1
        self.this_accept_num += 1

    def update_accept_fast_charge_num(self):
        self.total_accept_fast_charge_num += 1
        self.this_accept_fast_charge_num += 1
        self.total_accept_num += 1
        self.this_accept_num += 1

    def update_accept_slow_charge_num(self):
        self.total_accept_slow_charge_num += 1
        self.this_accept_slow_charge_num += 1
        self.total_accept_num += 1
        self.this_accept_num += 1

    def get_removed_park_num(self):
        self.removed_park_num += 1

    def get_removed_fast_charge_num(self):
        self.removed_fast_charge_num += 1

    def get_removed_slow_charge_num(self):
        self.removed_slow_charge_num += 1

    def update_available_os_num(self):
        self.available_os -= self.this_accept_park_num + self.removed_park_num

    def update_available_fcs_num(self):
        self.available_fcs -= self.this_accept_num + self.removed_fast_charge_num

    def update_available_scs_num(self):
        self.available_scs -= self.this_accept_slow_charge_num + self.removed_slow_charge_num

    def update_occupancy(self):
        self.occupancy = (self.total_slot - self.available_fcs - self.available_fcs - self.available_scs) / self.total_slot


class Slot(object):
    def __int__(
            self,
            id,
            parkinglot_id,
    ):
        self.id = id
        self.parkinglot_id = parkinglot_id
        self.occupied_time = np.zeros((1440, 1)),  # 当满足某个需求后 更新该值


class OrdinarySlot(Slot):
    def __int__(
            self,
            id,
            parkinglot_id,
    ):
        super(OrdinarySlot, self).__int__(id, parkinglot_id)


class FastChargeSlot(Slot):
    def __int__(
            self,
            id,
            parkinglot_id,
    ):
        super(FastChargeSlot, self).__int__(id, parkinglot_id)


class SlowChargeSlot(Slot):
    def __int__(
            self,
            id,
            parkinglot_id,
    ):
        super(SlowChargeSlot, self).__int__(id, parkinglot_id)


def update_occupied_time(slot: Slot, demand: Demand):
    # 1440*1的array
    if demand.charge_label == 0:
        return slot.occupied_time + demand.parking_t
    else:
        return slot.occupied_time + demand.char_t
