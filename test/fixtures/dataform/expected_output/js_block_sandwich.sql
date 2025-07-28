



with regular as (
    select * from `project.dataset.table_a`
),
templated_str as (
    select * from `project.dataset.table_b`
)
select *
from regular
    inner join templated_str
        on regular.some_id = templated_str.some_id
where regular.some_column = 'config_js_query'



    and regular.other_column = 'after_second_js_block'
