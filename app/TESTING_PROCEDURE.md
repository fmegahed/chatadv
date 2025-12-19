# Local test

1. Place `chatadv.php` in a PHP-enabled directory with the `.env` file and the `system_prompt.txt`

2. From the same directory start a cmd.exe and start thePHP’s built-in server via:  `php -S 127.0.0.1:8080`

3. In a different cmd window, test with: curl "http://127.0.0.1:8080/chatadv.php?question=What%20are%20the%20prerequisites%20for%20ISA%20401?"


Expected result: JSON containing an `answer` field and `citation` text.