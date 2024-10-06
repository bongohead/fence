library(tidyverse)

expand_grid(
	x = 1:100,
	y = 1:50
) %>%
	mutate(
		z = rnorm(nrow(.), 0, 1)
	) %>%
	ggplot() +
	geom_tile(aes(x = x, y = y, fill = z)) +
	scale_fill_gradientn(
		colors = c("red", 'orange', "yellow", "green", 'lawngreen'),
		values = c(0, 0.4, .6, .8, 1)
	) +
	theme_minimal() +
	theme(
		axis.text = element_blank(),    # Hide axis text (tick labels)
		axis.ticks = element_blank(),   # Hide axis ticks
		panel.grid = element_blank(),   # Hide gridlines
		panel.border = element_blank()  # Hide the border
	) +
	guides(fill = "none") +
	labs(x = 'Hidden State Dimension', y = 'Token')


expand_grid(
	x = 1:100,
	y = 1:50
) %>%
	mutate(
		z = ifelse(
			!x %in% 80:85,
			rnorm(nrow(.), 0, 1),
			rnorm(nrow(.), 2, .8)
		)
	) %>%
	ggplot() +
	geom_tile(aes(x = x, y = y, fill = z)) +
	scale_fill_gradientn(
		colors = c("red", 'orange', "yellow", "green"),
		values = c(0, 0.4, .6, 1)
	) +
	theme_minimal() +
	theme(
		axis.text = element_blank(),    # Hide axis text (tick labels)
		axis.ticks = element_blank(),   # Hide axis ticks
		panel.grid = element_blank(),   # Hide gridlines
		panel.border = element_blank()  # Hide the border
	) +
	guides(fill = "none") +
	labs(x = 'Hidden State Dimension', y = 'Token')


expand_grid(
	x = 1:100,
	y = 1:50
) %>%
	mutate(
		z = ifelse(
			!x %in% 60:65,
			rnorm(nrow(.), 0, 1),
			rnorm(nrow(.), 2, .8)
		)
	) %>%
	ggplot() +
	geom_tile(aes(x = x, y = y, fill = z)) +
	scale_fill_gradientn(
		colors = c("red", 'orange', "yellow", "green"),
		values = c(0, 0.4, .6, 1)
	) +
	theme_minimal() +
	theme(
		axis.text = element_blank(),    # Hide axis text (tick labels)
		axis.ticks = element_blank(),   # Hide axis ticks
		panel.grid = element_blank(),   # Hide gridlines
		panel.border = element_blank()  # Hide the border
	) +
	guides(fill = "none") +
	labs(x = 'Hidden State Dimension', y = 'Token')


expand_grid(
	x = 1:100,
	y = 1:50
) %>%
	mutate(
		z = ifelse(
			!x %in% c(60:65, 80:85),
			rnorm(nrow(.), 0, 1),
			rnorm(nrow(.), 2, .8)
		)
	) %>%
	ggplot() +
	geom_tile(aes(x = x, y = y, fill = z)) +
	scale_fill_gradientn(
		colors = c("red", 'orange', "yellow", "green"),
		values = c(0, 0.4, .6, 1)
	) +
	theme_minimal() +
	theme(
		axis.text = element_blank(),    # Hide axis text (tick labels)
		axis.ticks = element_blank(),   # Hide axis ticks
		panel.grid = element_blank(),   # Hide gridlines
		panel.border = element_blank()  # Hide the border
	) +
	guides(fill = "none") +
	labs(x = 'Hidden State Dimension', y = 'Token')



