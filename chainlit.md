# To start, you can ask something like:

*List me all users with their profile picture please* 🍇🤖


* The bot will try to do it's best and answer with a well written and
formatted response!

```sql
SELECT users_user.id, users_profile.image
FROM users_user
JOIN users_profile
ON users_user.id = users_profile.user_id;
```

* You can copy the query and ask for the bot to improve it.

* Run the query on your own database/workbench.

* Click the RUN action button and wait for results!
