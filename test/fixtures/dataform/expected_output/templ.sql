



with example_data as (
  select * from `project.dataset.source`
)
select * from example_data
where label = 'target_a'
and

other_lable = 'target_other'
