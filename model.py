import pandas as pd
from pulp import *
import ipdb
import cplex
from profilehooks import timecall
import matplotlib.pyplot as plt


class Model(object):
    """docstring for Model"""

    def __init__(self):
        try:
            self.outbound_dist = pd.read_csv("./CSV/outbound_dist.csv",
                                             index_col=[0])
            self.deliv_orders = pd.read_csv("./CSV/ddu_orders.csv",
                                            index_col=[0],
                                            nrows=50)
            self.pu_orders = pd.read_csv("./CSV/exw_orders.csv",
                                         index_col=[0],
                                         nrows=50)
            self.deliv_ttkm = pd.read_csv("./CSV/ddu_TTKM.csv",
                                          index_col=[0],
                                          nrows=50)
            self.pu_ttkm = pd.read_csv("./CSV/exw_TTKM.csv",
                                       index_col=[0])
            self.deliv_orders_count = pd.read_csv("./CSV/ddu_TTKM.csv",
                                                  index_col=[0],
                                                  nrows=50)
            self.warehouse_data = pd.read_csv("./CSV/warehouse_costs.csv",
                                              index_col=[0])
            self.inbound_dist = pd.read_csv("./CSV/inbound_dist.csv",
                                            index_col=[0])
            self.deliv_orders_data = pd.read_csv("./CSV/ddu_orders.csv",
                                                 index_col=[0])
            self.centralization = 19
            self.opt_model = pulp.LpProblem("Network_Optimization",
                                            LpMinimize)
            self.deliv_orders = self.deliv_orders[(
                self.deliv_orders.T != 0).any()]
            self.pu_orders = self.pu_orders[(self.pu_orders.T != 0).any()]

            self.outbound_dist = self.outbound_dist[(self.outbound_dist.index.isin(
                self.deliv_orders.index) | self.outbound_dist.index.isin(self.pu_orders.index))]

        except IOError as e:
            print("file error")

    def define_indices(self):
        self.wh_id = self.warehouse_data.index.tolist()
        self.deliv_client_id = self.deliv_orders.index.tolist()
        self.pu_client_id = self.pu_orders.index.tolist()
        self.datetime = list(self.deliv_orders.columns)
        self.prod_facilities = self.inbound_dist.index.tolist()
        self.buffer_facilities = list(self.inbound_dist.columns)

    def define_variables(self):
        self.deliv_flow = pulp.LpVariable.dicts("deliv flow",
                                                (self.deliv_client_id,
                                                 self.wh_id, self.datetime),
                                                lowBound=0)
        self.pu_flow = pulp.LpVariable.dicts("pick-up flow",
                                             (self.pu_client_id,
                                              self.wh_id, self.datetime),
                                             lowBound=0)
        self.wh_open = pulp.LpVariable.dicts("warehouse opened",
                                             self.wh_id,
                                             cat="Binary")
        self.backlog_deliver = pulp.LpVariable.dicts("delivery backlog",
                                                     (self.deliv_client_id,
                                                      self.wh_id, self.datetime),
                                                     lowBound=0)
        self.backlog_pu = pulp.LpVariable.dicts("pick-up backlog",
                                                (self.pu_client_id,
                                                 self.wh_id, self.datetime),
                                                lowBound=0)
        self.inbound_flow = pulp.LpVariable.dicts("inbound flow",
                                                  (self.prod_facilities,
                                                   self.buffer_facilities,
                                                   self.datetime),
                                                  lowBound=0)
        self.extra_capacity = pulp.LpVariable.dicts("extra capaicty",
                                                    (self.wh_id,
                                                        self.datetime),
                                                    lowBound=0)

    @timecall
    def define_objective(self):
        # fixed costs of opening warehouses

        fixed_c = pulp.lpSum(
            (self.wh_open[w] * self.warehouse_data.loc[w, "fixed costs"]
                for w in self.wh_id))
        deliv_var_c = pulp.lpSum(
            (self.deliv_flow[c][w][t] * self.warehouse_data.loc[w, "variable costs"]
             for c in self.deliv_client_id
             for w in self.wh_id
             for t in self.datetime))
        pu_var_c = pulp.lpSum(
            (self.pu_flow[c][w][t] * self.warehouse_data.loc[w, "variable costs"]
             for c in self.pu_client_id
             for w in self.wh_id
             for t in self.datetime))  # costs of handling pick-up orders
        backlog_c = pulp.lpSum(
            ((self.backlog_deliver[c][w][t] + self.backlog_pu[i][w][t]) * self.warehouse_data.loc[w, "backlog"]
             for c in self.deliv_client_id
             for i in self.pu_client_id
             for w in self.wh_id
             for t in self.datetime))
        # + pulp.lpSum(
        #     [(self.inbound_flow[i][j][t] * (1/30)) * self.inbound_dist.loc[i, j]
        #      for i in self.prod_facilities
        #      for j in self.buffer_facilities
        #      for t in self.datetime])
        trans_c = pulp.lpSum(
            (self.deliv_ttkm.loc[c, t] * self.outbound_dist.loc[c, w] + self.deliv_orders_count.loc[c, t] * 32.5
             for c in self.deliv_client_id
             for w in self.wh_id
             for t in self.datetime
             if self.deliv_flow[c][w][t] >= 0.001))
        extra_cap_c = pulp.lpSum(
            (self.extra_capacity[w][t] * 150
             for w in self.wh_id
             for t in self.datetime))
        self.opt_model += fixed_c + deliv_var_c + \
            pu_var_c + backlog_c + trans_c + extra_cap_c

    @timecall
    def define_constraints(self):
        # Demand Constraint
        for t in self.datetime:
            for c in self.deliv_client_id:
                # demand delivery
                self.opt_model += pulp.lpSum((self.deliv_flow[c][w][t] + self.backlog_deliver[c][w][t]
                                              for w in self.wh_id
                                              if self.deliv_orders.loc[c, t] > 0
                                              )) == self.deliv_orders.loc[c, t]
            for c in self.pu_client_id:
                # demand pick-up
                self.opt_model += pulp.lpSum((self.pu_flow[c][w][t] + self.backlog_pu[c][w][t]
                                              for w in self.wh_id
                                              if self.pu_orders.loc[c, t] > 0
                                              )) == self.pu_orders.loc[c, t]
            # ipdb.set_trace()
        for ind, t in enumerate(self.datetime):
            if ind != 0:

                for w in self.wh_id:
                    # capacity
                    td = self.datetime[ind - 1]
                    self.opt_model += pulp.lpSum((self.deliv_flow[c][w][t] + self.pu_flow[i][w][t] +
                                                  self.backlog_deliver[c][w][td] +
                                                  self.backlog_pu[i][w][td]
                                                  for c in self.deliv_client_id
                                                  for i in self.pu_client_id)) <= (self.warehouse_data.loc[w, "capacity"] * self.wh_open[w]) + self.extra_capacity[w][t]
            else:
                for w in self.wh_id:
                    # capacity first time period
                    self.opt_model += pulp.lpSum((self.deliv_flow[c][w][t] + self.pu_flow[i][w][t]
                                                  for c in self.deliv_client_id
                                                  for i in self.pu_client_id)) <= (self.warehouse_data.loc[w, "capacity"] * self.wh_open[w]) + self.extra_capacity[w][t]

        self.opt_model += pulp.lpSum((self.wh_open[w]
                                      for w in self.wh_id)) <= self.centralization

    def trans_costs(self, ttkm, distance, count, fixum, order):
        if order == 0:
            return 0
        print(ttkm * distance + count * fixum)
        return ttkm * distance + count * fixum

    def post_process_backlog(self, backlog1, backlog2):
        temp = [backlog1, backlog2]
        dic = {"client": [], "warehouse": [], "date": [], "value": []}
        ipdb.set_trace()
        for j in range(0, 2):
            for c in temp[j][1]:
                for w in self.wh_id:
                    if self.wh_open[w].varValue == 1:
                        for t in self.datetime:
                            dic["client"].append(c)
                            dic["warehouse"].append(w)
                            dic["date"].append(t)
                            dic["value"].append(temp[j][0][c][w][t].varValue)
                    # ipdb.set_trace()

        self.output = pd.DataFrame.from_dict(dic)
        print("backlog: ")
        print(self.output.groupby(["date", "warehouse"]).sum())

        self.output.to_csv("backlog.csv",
                           encoding="utf-8",
                           index=False)

    def export_flows(self, variable, client_id):
        self.output = pd.DataFrame()
        dic = {"client": [], "warehouse": [], "date": [], "value": []}
        for c in client_id:
            for w in self.wh_id:
                if self.wh_open[w].varValue == 1:
                    for t in self.datetime:
                        dic["client"].append(c)
                        dic["warehouse"].append(w)
                        dic["date"].append(t)
                        dic["value"].append(variable[c][w][t].varValue)
                    # ipdb.set_trace()
        self.output = pd.DataFrame.from_dict(dic)

        if variable == self.deliv_flow:
            print("deliv: ", self.output)
            self.output.to_csv("delivery_flows" + ".csv",
                               encoding="utf-8",
                               index=False)
        if variable == self.pu_flow:
            print("pick-up: ", self.output)
            self.output.to_csv("pick-up_flows" + ".csv",
                               encoding="utf-8",
                               index=False)
        return self.output

    def export_slack(self):
        dic = {"warehouse": [], "date": [], "value": []}
        for w in self.wh_id:
            if self.wh_open[w].varValue == 1:
                for t in self.datetime:
                    dic["warehouse"].append(w)
                    dic["date"].append(t)
                    dic["value"].append(self.extra_capacity[w][t].varValue)

        self.slack_output = pd.DataFrame.from_dict(dic)
        print(self.slack_output)

    @timecall
    def build_model(self):
        self.define_indices()
        self.define_variables()
        self.define_objective()
        self.define_constraints()
        solver = CPLEX_PY()
        self.opt_model.solve(solver)
        c = cplex.Cplex()
        c.conflict.refine
        for w in self.wh_id:
            print(self.wh_open[w],
                  self.wh_open[w].varValue)

        # self.deliv_flow_sol = pd.DataFrame()
        # for v in self.opt_model.variables():
        #     print(v.name, "=", v.varValue)

        self.post_process_backlog((self.backlog_deliver, self.deliv_client_id),
                                  (self.backlog_pu, self.pu_client_id))
        self.export_flows(self.deliv_flow, self.deliv_client_id)
        print("deliv: ")
        print(self.output.groupby(["date", "warehouse"]).sum())
        self.export_flows(self.pu_flow, self.pu_client_id)
        print("pick-up: ")
        print(self.output.groupby(["date", "warehouse"]).sum())
        self.export_slack()
        ipdb.set_trace()

    def graph_variable(self):
        self.deliv_df = pd.read_csv("delivery_flows.csv")
        self.pu_df = pd.read_csv("pick-up_flows.csv")
        self.backlog_df = pd.read_csv("backlog.csv")

        deliv_wh_agg = self.deliv_df.groupby(by=["warehouse"]).sum()
        deliv_wh_agg.plot.bar(title="Delivery orders per warehouse")
        plt.show()

        pu_wh_agg = self.pu_df.groupby(by=["warehouse"]).sum()
        pu_wh_agg.plot.bar(title="Pick-up order per warehouse")
        plt.show()

        bl_date_agg = self.backlog_df.groupby(by=["date"]).sum()
        bl_date_agg.plot.bar(title="Backloged orders per month")
        plt.show()


a = Model()
a.build_model()
a.graph_variable()
