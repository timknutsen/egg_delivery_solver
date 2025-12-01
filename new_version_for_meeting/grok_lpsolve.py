library(lpSolve)

# Example data
orders <- data.frame(
  Ordernr = c(1, 2, 3, 4, 5),
  OrderStatus = c("Bekreftet", "Bekreftet", "Bekreftet", "Bekreftet", "Bekreftet"),
  DeliveryDate = as.Date(c("2025-01-05", "2025-01-20", "2025-02-01", "2025-01-10", "2025-01-25")),
  Product = c("Gain", "Shield", "Gain", "Shield", "Gain"),
  Organic = c("Yes", "No", "No", "Yes", "Yes"),
  Volume = c(300, 400, 500, 200, 600),
  PreferredSite = c("Gruppe A", "Gruppe B", "Gruppe C", "Gruppe A", "Gruppe B")
)

fish_groups <- data.frame(
  Site = c("Gruppe A", "Gruppe B", "Gruppe C", "dummy"),
  StrippingStopDate = as.Date(c("2025-01-01", "2025-01-15", "2025-01-30", "1900-01-01")),
  GainEggs = c(1000, 800, 1200, Inf),
  ShieldEggs = c(500, 600, 700, Inf),
  Organic = c("Yes", "No", "Yes", "No")
)

# Filter out cancelled orders
orders <- orders[orders$OrderStatus != "Kansellert", ]

# Define sets
O <- orders$Ordernr
G <- fish_groups$Site
G_real <- G[G != "dummy"]

# Helper functions to get parameters
get_param <- function(df, col, index = "Ordernr") {
  setNames(df[[col]], df[[index]])
}
delivery_date <- get_param(orders, "DeliveryDate")
product <- get_param(orders, "Product")
organic_req <- get_param(orders, "Organic")
volume <- get_param(orders, "Volume")
preferred_site <- get_param(orders, "PreferredSite")

stripping_stop <- get_param(fish_groups, "StrippingStopDate", "Site")
gain_eggs <- get_param(fish_groups, "GainEggs", "Site")
shield_eggs <- get_param(fish_groups, "ShieldEggs", "Site")
organic_group <- get_param(fish_groups, "Organic", "Site")

# Build variable list and constraints
variables <- list()
var_names <- c()
costs <- c()
constr_matrix <- list()
constr_dir <- c()
constr_rhs <- c()

var_index <- 1
for (o in O) {
  if (product[o] == "Gain") {
    possible_g <- G_real[
      stripping_stop[G_real] <= delivery_date[o] &
      (organic_req[o] == "No" | organic_group[G_real] == "Yes") &
      gain_eggs[G_real] >= volume[o]
    ]
    for (g in possible_g) {
      var_names <- c(var_names, paste0("x_", o, "_", g))
      costs <- c(costs, ifelse(g == preferred_site[o], 0, 1))
      variables[[paste0("x_", o, "_", g)]] <- list(o = o, g = g, type = "Gain")
    }
    var_names <- c(var_names, paste0("d_", o))
    costs <- c(costs, 2)
    variables[[paste0("d_", o)]] <- list(o = o, g = "dummy", type = "dummy")
  } else if (product[o] == "Shield") {
    possible_g_shield <- G_real[
      stripping_stop[G_real] <= delivery_date[o] &
      (organic_req[o] == "No" | organic_group[G_real] == "Yes") &
      shield_eggs[G_real] >= volume[o]
    ]
    possible_g_gain <- G_real[
      stripping_stop[G_real] <= delivery_date[o] &
      (organic_req[o] == "No" | organic_group[G_real] == "Yes") &
      gain_eggs[G_real] >= volume[o]
    ]
    for (g in possible_g_shield) {
      var_names <- c(var_names, paste0("y_", o, "_", g))
      costs <- c(costs, ifelse(g == preferred_site[o], 0, 1))
      variables[[paste0("y_", o, "_", g)]] <- list(o = o, g = g, type = "Shield")
    }
    for (g in possible_g_gain) {
      var_names <- c(var_names, paste0("z_", o, "_", g))
      costs <- c(costs, ifelse(g == preferred_site[o], 0, 1))
      variables[[paste0("z_", o, "_", g)]] <- list(o = o, g = g, type = "Gain")
    }
    var_names <- c(var_names, paste0("d_", o))
    costs <- c(costs, 2)
    variables[[paste0("d_", o)]] <- list(o = o, g = "dummy", type = "dummy")
  }
}

# Assignment constraints
for (o in O) {
  row <- numeric(length(var_names))
  for (i in seq_along(var_names)) {
    v <- variables[[var_names[i]]]
    if (v$o == o) row[i] <- 1
  }
  constr_matrix[[length(constr_matrix) + 1]] <- row
  constr_dir <- c(constr_dir, "=")
  constr_rhs <- c(constr_rhs, 1)
}

# Capacity constraints
for (g in G_real) {
  # GainEggs
  gain_row <- numeric(length(var_names))
  for (i in seq_along(var_names)) {
    v <- variables[[var_names[i]]]
    if (v$g == g && (v$type == "Gain")) {
      gain_row[i] <- volume[v$o]
    }
  }
  constr_matrix[[length(constr_matrix) + 1]] <- gain_row
  constr_dir <- c(constr_dir, "<=")
  constr_rhs <- c(constr_rhs, gain_eggs[g])
  
  # ShieldEggs
  shield_row <- numeric(length(var_names))
  for (i in seq_along(var_names)) {
    v <- variables[[var_names[i]]]
    if (v$g == g && v$type == "Shield") {
      shield_row[i] <- volume[v$o]
    }
  }
  constr_matrix[[length(constr_matrix) + 1]] <- shield_row
  constr_dir <- c(constr_dir, "<=")
  constr_rhs <- c(constr_rhs, shield_eggs[g])
}

# Convert constraints to matrix
constr_matrix <- do.call(rbind, constr_matrix)

# Solve the model
result <- lp("min", costs, constr_matrix, constr_dir, constr_rhs, all.bin = TRUE)

# Extract solution
if (result$status == 0) {
  cat("Optimal solution found:\n")
  solution <- result$solution
  for (i in seq_along(var_names)) {
    if (solution[i] > 0.5) {
      v <- variables[[var_names[i]]]
      cat(sprintf("Order %d: Assigned to %s using %s\n", v$o, v$g, ifelse(v$type == "dummy", "dummy", v$type)))
    }
  }
} else {
  cat("No optimal solution found.\n")
}